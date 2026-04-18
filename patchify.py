import concurrent.futures
import math
import os
import random
import shutil

# Reserve one logical core for the QGIS UI thread and OS scheduler.
# The default starts at half the available cores — conservative enough not to
# overwhelm the machine on first use, yet still meaningfully parallel.
_CPU_COUNT       = os.cpu_count() or 1
_MAX_WORKERS     = max(1, _CPU_COUNT - 1)
_DEFAULT_WORKERS = max(1, _CPU_COUNT // 2)

import numpy as np
from osgeo import gdal, gdal_array, ogr, osr
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QSpinBox, QToolButton, QVBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel, QgsWkbTypes
from .augment import augment


_EXT_TO_DRIVER = {'tif': 'GTiff', 'tiff': 'GTiff', 'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'}


def _image_info(filepath):
    """Return a one-line metadata string for an image without reading pixel data."""
    ds = gdal.Open(filepath)
    if ds is None:
        return ''
    w, h   = ds.RasterXSize, ds.RasterYSize
    bands  = ds.RasterCount
    dtype  = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    gt     = ds.GetGeoTransform()
    band_s = f'{bands} band{"s" if bands > 1 else ""}'
    px_s   = ''
    # Only show pixel size when the image has a real georeference (not default identity)
    if gt and not (gt[0] == 0.0 and gt[1] == 1.0 and gt[3] == 0.0):
        pw, ph = abs(gt[1]), abs(gt[5])
        px_s = f'  |  {pw:.4g} × {ph:.4g} m/px' if pw != ph else f'  |  {pw:.4g} m/px'
    return f'{w:,} × {h:,} px  |  {band_s}  |  {dtype}{px_s}'


def _vector_mask_info(layer):
    """Return a one-line metadata string for a vector mask layer."""
    geom_s     = QgsWkbTypes.displayString(layer.wkbType())
    count      = layer.featureCount()
    num_fields = [f.name() for f in layer.fields() if f.isNumeric()]
    if num_fields:
        listed = ', '.join(num_fields[:6]) + ('…' if len(num_fields) > 6 else '')
        f_s = f'Numeric fields: {listed}'
    else:
        f_s = 'No numeric fields — burn value will be used'
    return f'{geom_s}  |  {count:,} features  |  {f_s}'


def _crs_mismatch(raster_layer, vector_layer):
    """Return a warning string if the two layers have different CRS, else empty string."""
    if raster_layer is None or vector_layer is None:
        return ''
    r_crs = raster_layer.crs()
    v_crs = vector_layer.crs()
    if r_crs.isValid() and v_crs.isValid() and r_crs != v_crs:
        return (
            f'CRS mismatch — mask ({v_crs.authid()}) differs from image '
            f'({r_crs.authid()}). Will be reprojected automatically.'
        )
    return ''


def _rasterize_tile(mask_source, mask_layer_name, tile_gt, projection,
                    tile_w, tile_h, burn_field=None, burn_value=1):
    """Rasterize only the vector features that intersect one tile.

    Each worker calls this independently — no shared state.  A spatial filter
    limits OGR to features that touch the tile bbox, so large vector datasets
    are not fully scanned per tile.  CRS reprojection is applied when needed.
    """
    vec_ds    = ogr.Open(mask_source)
    if vec_ds is None:
        return None
    ogr_layer = (vec_ds.GetLayerByName(mask_layer_name)
                 if mask_layer_name else vec_ds.GetLayer())
    if ogr_layer is None:
        return None

    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(projection)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src_srs = ogr_layer.GetSpatialRef()

    # Tile bounding box in raster CRS (tile_gt[5] is negative → y_min < y_max)
    x_min = tile_gt[0]
    x_max = tile_gt[0] + tile_w * tile_gt[1]
    y_max = tile_gt[3]
    y_min = tile_gt[3] + tile_h * tile_gt[5]

    if src_srs is not None and not src_srs.IsSame(target_srs):
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        # Transform tile bbox to vector CRS for the spatial pre-filter
        inv_t   = osr.CoordinateTransformation(target_srs, src_srs)
        corners = [inv_t.TransformPoint(x, y)[:2]
                   for x, y in ((x_min, y_min), (x_min, y_max),
                                (x_max, y_min), (x_max, y_max))]
        xs, ys  = zip(*corners)
        ogr_layer.SetSpatialFilterRect(min(xs), min(ys), max(xs), max(ys))

        # Reproject filtered features into raster CRS
        fwd_t       = osr.CoordinateTransformation(src_srs, target_srs)
        mem_ogr_drv = ogr.GetDriverByName('Memory')
        mem_ogr_ds  = mem_ogr_drv.CreateDataSource('')
        mem_layer   = mem_ogr_ds.CreateLayer('', srs=target_srs,
                                              geom_type=ogr_layer.GetGeomType())
        layer_defn  = ogr_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            mem_layer.CreateField(layer_defn.GetFieldDefn(i))
        out_defn = mem_layer.GetLayerDefn()
        for feat in ogr_layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            geom = geom.Clone()
            geom.Transform(fwd_t)
            new_feat = ogr.Feature(out_defn)
            new_feat.SetGeometry(geom)
            for i in range(out_defn.GetFieldCount()):
                new_feat.SetField(i, feat.GetField(i))
            mem_layer.CreateFeature(new_feat)
        ogr_layer = mem_layer
    else:
        ogr_layer.SetSpatialFilterRect(x_min, y_min, x_max, y_max)

    mem_drv = gdal.GetDriverByName('MEM')
    out_ds  = mem_drv.Create('', tile_w, tile_h, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(tile_gt)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).Fill(0)

    opts  = [f'ATTRIBUTE={burn_field}'] if burn_field else []
    burns = [] if burn_field else [burn_value]
    gdal.RasterizeLayer(out_ds, [1], ogr_layer, burn_values=burns, options=opts)
    out_ds.FlushCache()
    return out_ds.GetRasterBand(1).ReadAsArray()


def _read_raster_meta(filepath):
    """Open a raster and return metadata without reading any pixel data."""
    ds = gdal.Open(filepath)
    if ds is None:
        return None
    return {
        'width':        ds.RasterXSize,
        'height':       ds.RasterYSize,
        'band_count':   ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection':   ds.GetProjection(),
        'nodata':       ds.GetRasterBand(1).GetNoDataValue(),
    }


def _save_patch(array, filepath, ext, geotransform=None, projection=None):
    array = np.ascontiguousarray(array)
    driver = gdal.GetDriverByName(_EXT_TO_DRIVER.get(ext.lower(), 'GTiff'))
    h, w = array.shape[:2]
    n_bands = 1 if array.ndim == 2 else array.shape[2]
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    ds = driver.Create(filepath, w, h, n_bands, dtype)
    if n_bands == 1:
        ds.GetRasterBand(1).WriteArray(array if array.ndim == 2 else array[:, :, 0])
    else:
        for b in range(n_bands):
            ds.GetRasterBand(b + 1).WriteArray(array[:, :, b])
    if geotransform is not None:
        ds.SetGeoTransform(geotransform)
    if projection:
        ds.SetProjection(projection)
    ds.FlushCache()
    ds = None


def _tile_geotransform(src_gt, x1, y1):
    return (
        src_gt[0] + x1 * src_gt[1] + y1 * src_gt[2],
        src_gt[1], src_gt[2],
        src_gt[3] + x1 * src_gt[4] + y1 * src_gt[5],
        src_gt[4], src_gt[5],
    )


def _process_tile(row, col, *, img_path, n_bands, nodata_value,
                  mask_source, mask_layer_name, mask_enabled, foreground_value,
                  burn_field, burn_value,
                  patchStep_y, patchStep_x, patchSize_y, patchSize_x,
                  selected_methods, aug_suffixes, outname, img_ext, mask_ext,
                  total_img_path, total_mask_path, steps_in_width,
                  georef, src_geotransform, src_projection):
    y1 = row * patchStep_y
    x1 = col * patchStep_x
    tile_num = row * steps_in_width + col + 1

    # Each worker opens its own GDAL handle and reads only its tile window
    ds    = gdal.Open(img_path)
    bands = [ds.GetRasterBand(i + 1).ReadAsArray(x1, y1, patchSize_x, patchSize_y)
             for i in range(n_bands)]
    crop  = bands[0] if n_bands == 1 else np.stack(bands, axis=-1)
    ds    = None  # release file handle

    # Skip tiles that are entirely nodata (e.g. black border after reprojection)
    if nodata_value is not None:
        all_nodata = (np.all(np.isnan(crop)) if math.isnan(nodata_value)
                      else np.all(crop == nodata_value))
        if all_nodata:
            return []

    tile_gt  = _tile_geotransform(src_geotransform, x1, y1)
    aug_imgs = augment(crop)

    aug_masks = None
    if mask_enabled:
        crop_mask = _rasterize_tile(
            mask_source, mask_layer_name, tile_gt, src_projection,
            patchSize_x, patchSize_y, burn_field, burn_value,
        )
        if crop_mask is None or not np.any(crop_mask == foreground_value):
            return []
        aug_masks = augment(crop_mask)

    geo_kw = {}
    if georef and src_geotransform is not None:
        geo_kw = {'geotransform': tile_gt, 'projection': src_projection}

    saved = []
    for idx in selected_methods:
        img_name = f"{tile_num}_{outname}_{aug_suffixes[idx]}.{img_ext}"
        _save_patch(aug_imgs[idx], os.path.join(total_img_path, img_name), img_ext, **geo_kw)
        saved.append(img_name)
        if mask_enabled and aug_masks is not None:
            msk_name = f"{tile_num}_{outname}_{aug_suffixes[idx]}.{mask_ext}"
            _save_patch(aug_masks[idx], os.path.join(total_mask_path, msk_name), mask_ext)

    return saved


# -----------------------------------------------------------------------------
# Plugin entry point
# -----------------------------------------------------------------------------

class PatchifyPlugin:
    def __init__(self, iface):
        self.iface  = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon = QIcon(os.path.join(os.path.dirname(__file__), 'crop_icon.png'))
        self.action = QAction(icon, 'Patchify', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu('Patchify', self.action)

    def unload(self):
        self.iface.removePluginRasterMenu('Patchify', self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None

    def run(self):
        if self.dialog is None:
            self.dialog = PatchifyDialog(self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()


# -----------------------------------------------------------------------------
# Main dialog
# -----------------------------------------------------------------------------

class PatchifyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Patchify')
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'crop_icon.png')))
        self.setMinimumWidth(680)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        # --- Title ---
        title_label = QLabel('Patchify App')
        title_label.setStyleSheet('font: 18pt "Segoe Print"; color: #2c6fad;')
        root.addWidget(title_label)

        # --- Source & Export ---
        lbl_style = 'font: 10pt "Microsoft JhengHei UI"; font-weight: bold;'

        gb_io = QGroupBox('Source && Export')
        io_layout = QGridLayout(gb_io)
        io_layout.setColumnStretch(1, 1)
        io_layout.setVerticalSpacing(6)
        io_layout.setHorizontalSpacing(8)

        info_style = 'font: 8pt "Microsoft JhengHei UI"; color: #2c6fad; font-style: italic;'

        # Input image — raster layers from QGIS layer panel
        label_input = QLabel('Input Image')
        label_input.setStyleSheet(lbl_style)
        self.combo_input = QgsMapLayerComboBox()
        self.combo_input.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.lbl_image_info = QLabel('')
        self.lbl_image_info.setStyleSheet(info_style)

        # Export folder — still a plain folder picker
        label_export = QLabel('Export Folder')
        label_export.setStyleSheet(lbl_style)
        self.lineEdit_export = QLineEdit()
        self.btn_export = QToolButton()
        self.btn_export.setText('...')

        # Mask row — vector layers from QGIS layer panel; disabled until checkbox ticked
        self.check_mask_enabled = QCheckBox('Mask / Label Image')
        self.check_mask_enabled.setStyleSheet(lbl_style)
        self.combo_mask = QgsMapLayerComboBox()
        self.combo_mask.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.combo_mask.setEnabled(False)
        # Burn-field selector — populated from the vector layer's numeric fields
        self.lbl_burn_field = QLabel('Attribute Field:')
        self.lbl_burn_field.setStyleSheet('font: 9pt "Microsoft JhengHei UI"; color: #444;')
        self.lbl_burn_field.setEnabled(False)
        self.combo_mask_field = QComboBox()
        self.combo_mask_field.setEnabled(False)
        self.combo_mask_field.setToolTip(
            'Field whose values are burned into the raster mask.\n'
            'Choose "— burn value —" to use the Binary Foreground Value for all features.'
        )
        self.lbl_mask_info = QLabel('Select a vector layer to use as mask / label')
        self.lbl_mask_info.setStyleSheet('font: 8pt "Microsoft JhengHei UI"; color: #666; font-style: italic;')

        # Foreground value — used to skip tiles with no foreground pixels (binary masks)
        self.lbl_foreground = QLabel('Binary Foreground Value (1–255):')
        self.lbl_foreground.setStyleSheet('font: 9pt "Microsoft JhengHei UI"; color: #444;')
        self.lbl_foreground.setEnabled(False)
        self.spin_foreground = QSpinBox()
        self.spin_foreground.setRange(1, 255)
        self.spin_foreground.setValue(255)
        self.spin_foreground.setMaximumWidth(60)
        self.spin_foreground.setEnabled(False)

        io_layout.addWidget(label_input,             0, 0)
        io_layout.addWidget(self.combo_input,        0, 1, 1, 2)
        io_layout.addWidget(self.lbl_image_info,     1, 0, 1, 3)
        io_layout.addWidget(label_export,            2, 0)
        io_layout.addWidget(self.lineEdit_export,    2, 1)
        io_layout.addWidget(self.btn_export,         2, 2)
        io_layout.addWidget(self.check_mask_enabled, 3, 0)
        io_layout.addWidget(self.combo_mask,         3, 1, 1, 2)
        io_layout.addWidget(self.lbl_burn_field,     4, 0)
        io_layout.addWidget(self.combo_mask_field,   4, 1, 1, 2)
        io_layout.addWidget(self.lbl_mask_info,      5, 0, 1, 3)
        io_layout.addWidget(self.lbl_foreground,     6, 0)
        io_layout.addWidget(self.spin_foreground,    6, 1)
        root.addWidget(gb_io)

        # --- Middle row: Patch Configuration + Dataset Split ---
        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)

        gb_crop = QGroupBox('Patch Configuration')
        crop_layout = QGridLayout(gb_crop)
        crop_layout.setVerticalSpacing(10)
        crop_layout.setHorizontalSpacing(6)

        self.lineEdit_winx = QLineEdit()
        self.lineEdit_winx.setMaximumWidth(55)
        self.lineEdit_winy = QLineEdit()
        self.lineEdit_winy.setMaximumWidth(55)
        self.lineEdit_stridex = QLineEdit()
        self.lineEdit_stridex.setMaximumWidth(55)
        self.lineEdit_stridey = QLineEdit()
        self.lineEdit_stridey.setMaximumWidth(55)

        crop_layout.addWidget(QLabel('Tile Size:'),       0, 0)
        crop_layout.addWidget(QLabel('X'),                0, 1)
        crop_layout.addWidget(self.lineEdit_winx,         0, 2)
        crop_layout.addWidget(QLabel('Y'),                0, 3)
        crop_layout.addWidget(self.lineEdit_winy,         0, 4)
        crop_layout.addWidget(QLabel('Stride:'),          1, 0)
        crop_layout.addWidget(QLabel('X'),                1, 1)
        crop_layout.addWidget(self.lineEdit_stridex,      1, 2)
        crop_layout.addWidget(QLabel('Y'),                1, 3)
        crop_layout.addWidget(self.lineEdit_stridey,      1, 4)

        self.radio_georef = QRadioButton('Georeferenced')
        self.radio_array  = QRadioButton('Array Only')
        self.radio_georef.setChecked(True)
        save_mode_row = QHBoxLayout()
        save_mode_row.addWidget(self.radio_georef)
        save_mode_row.addWidget(self.radio_array)
        save_mode_row.addStretch()
        crop_layout.addWidget(QLabel('Save As:'), 2, 0)
        crop_layout.addLayout(save_mode_row,      2, 1, 1, 4)

        self.combo_format = QComboBox()
        self.combo_format.addItem('GeoTIFF (.tif)', 'tif')
        self.combo_format.addItem('PNG (.png)',      'png')
        self.combo_format.addItem('JPEG (.jpg)',     'jpg')
        crop_layout.addWidget(QLabel('Format:'), 3, 0)
        crop_layout.addWidget(self.combo_format, 3, 1, 1, 4)

        sep_crop = QFrame()
        sep_crop.setFrameShape(QFrame.HLine)
        sep_crop.setFrameShadow(QFrame.Sunken)
        crop_layout.addWidget(sep_crop, 4, 0, 1, 5)

        self.lbl_tile_count = QLabel('')
        self.lbl_tile_count.setStyleSheet(
            'font: 8pt "Microsoft JhengHei UI"; color: #2a7a2a; font-weight: bold;'
        )
        self.lbl_tile_count.setWordWrap(True)
        crop_layout.addWidget(self.lbl_tile_count, 5, 0, 1, 5)
        crop_layout.setRowStretch(6, 1)
        mid_row.addWidget(gb_crop)

        gb_output = QGroupBox('Dataset Split')
        out_layout = QGridLayout(gb_output)
        out_layout.setColumnStretch(1, 1)
        out_layout.setVerticalSpacing(8)
        self.lineEdit_outname = QLineEdit()
        self.lineEdit_train   = QLineEdit()
        self.lineEdit_train.setMaximumWidth(60)
        self.lineEdit_train.setText('70')
        self.lineEdit_test    = QLineEdit()
        self.lineEdit_test.setMaximumWidth(60)
        self.lineEdit_test.setText('20')
        self.lineEdit_valid   = QLineEdit()
        self.lineEdit_valid.setMaximumWidth(60)
        self.lineEdit_valid.setText('10')

        self.lbl_split_sum = QLabel()
        self.lbl_split_sum.setStyleSheet('font: 8pt "Microsoft JhengHei UI";')

        out_layout.addWidget(QLabel('Base Name:'),            0, 0)
        out_layout.addWidget(self.lineEdit_outname,           0, 1)
        out_layout.addWidget(QLabel('Training %'),            1, 0)
        out_layout.addWidget(self.lineEdit_train,             1, 1)
        out_layout.addWidget(QLabel('Testing %'),             2, 0)
        out_layout.addWidget(self.lineEdit_test,              2, 1)
        out_layout.addWidget(QLabel('Validation %'),          3, 0)
        out_layout.addWidget(self.lineEdit_valid,             3, 1)
        out_layout.addWidget(self.lbl_split_sum,              4, 0, 1, 2)
        mid_row.addWidget(gb_output)
        root.addLayout(mid_row)

        # --- Data Augmentation ---
        gb_aug = QGroupBox('Data Augmentation')
        aug_outer = QHBoxLayout(gb_aug)
        aug_outer.setSpacing(12)

        aug_left = QVBoxLayout()
        aug_left.setSpacing(8)
        self.radio_all    = QRadioButton('All')
        self.radio_custom = QRadioButton('Custom Selection')
        aug_left.addWidget(self.radio_all)
        aug_left.addWidget(self.radio_custom)
        aug_left.addStretch()

        sep_aug = QFrame()
        sep_aug.setFrameShape(QFrame.VLine)
        sep_aug.setFrameShadow(QFrame.Sunken)

        aug_right = QGridLayout()
        aug_right.setHorizontalSpacing(20)
        aug_right.setVerticalSpacing(6)
        self.check_original  = QCheckBox('Original Image')
        self.check_rotate90  = QCheckBox('Rotate 90°')
        self.check_rotate180 = QCheckBox('Rotate 180°')
        self.check_rotate270 = QCheckBox('Rotate 270°')
        self.check_flipv     = QCheckBox('Flip Vertically')
        self.check_fliph     = QCheckBox('Flip Horizontally')
        self.check_flipvh    = QCheckBox('Flip Vertically && Horizontally')
        aug_right.addWidget(self.check_original,  0, 0)
        aug_right.addWidget(self.check_rotate90,  1, 0)
        aug_right.addWidget(self.check_rotate180, 2, 0)
        aug_right.addWidget(self.check_rotate270, 3, 0)
        aug_right.addWidget(self.check_flipv,     1, 1)
        aug_right.addWidget(self.check_fliph,     2, 1)
        aug_right.addWidget(self.check_flipvh,    3, 1)
        aug_right.setColumnStretch(0, 1)
        aug_right.setColumnStretch(1, 1)

        aug_outer.addLayout(aug_left)
        aug_outer.addWidget(sep_aug)
        aug_outer.addLayout(aug_right, 1)
        root.addWidget(gb_aug)

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        root.addWidget(self.progress_bar)

        # --- Status label + Workers ---
        bottom_row = QHBoxLayout()
        self.progress_label = QLabel('Patchify is waiting for orders!')
        self.progress_label.setStyleSheet('font: 8pt "Microsoft JhengHei UI"; color: #444;')
        bottom_row.addWidget(self.progress_label, 1)
        bottom_row.addWidget(QLabel('Workers:'))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, _MAX_WORKERS)
        self.spin_workers.setValue(_DEFAULT_WORKERS)
        self.spin_workers.setMaximumWidth(55)
        self.spin_workers.setToolTip(
            f'Parallel threads for tile processing.\n'
            f'Max {_MAX_WORKERS} of {_CPU_COUNT} logical cores '
            f'(1 reserved for QGIS).\n'
            f'Default: {_DEFAULT_WORKERS} (half available).'
        )
        bottom_row.addWidget(self.spin_workers)
        root.addLayout(bottom_row)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_start  = QPushButton('Start Patching')
        self.btn_cancel = QPushButton('Close')
        self.btn_start.setMinimumWidth(110)
        self.btn_cancel.setMinimumWidth(80)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        root.addLayout(btn_row)

        # --- Signals & Slots ---
        self.radio_all.toggled.connect(self.allChecked)
        self.radio_custom.toggled.connect(self.customChecked)
        self.radio_georef.toggled.connect(self.updateFormatOptions)
        self.radio_array.toggled.connect(self.updateFormatOptions)
        self.btn_cancel.clicked.connect(self.onCancelClicked)
        self.combo_input.layerChanged.connect(self.onImageLayerChanged)
        self.btn_export.clicked.connect(self.pickSavingFolder)
        self.combo_mask.layerChanged.connect(self.onMaskLayerChanged)
        self.check_mask_enabled.toggled.connect(self.toggleMaskInput)
        self.btn_start.clicked.connect(self.patchifying)
        for field in (self.lineEdit_train, self.lineEdit_test, self.lineEdit_valid):
            field.textChanged.connect(self.updateSplitSum)
        self.updateSplitSum()
        self.lineEdit_winx.textChanged.connect(
            lambda v: self.lineEdit_stridex.setText(v) if not self.lineEdit_stridex.text() else None
        )
        self.lineEdit_winy.textChanged.connect(
            lambda v: self.lineEdit_stridey.setText(v) if not self.lineEdit_stridey.text() else None
        )
        for field in (self.lineEdit_winx, self.lineEdit_winy,
                      self.lineEdit_stridex, self.lineEdit_stridey):
            field.textChanged.connect(self.updateTileCount)
        for cb in (self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph,
                   self.check_flipvh):
            cb.stateChanged.connect(self.updateTileCount)
        # Seed info labels from whatever layer is already selected on open
        self.onImageLayerChanged(self.combo_input.currentLayer())

        self.updateFormatOptions()

        # Default augmentation state
        self.radio_all.setChecked(True)
        self.allChecked()

        # Default values
        self.Train_val            = 0
        self.Test_val             = 0
        self.Valid_val            = 0
        self.list_of_saved_names  = []
        self.mask_filename        = None
        self.maskNamePassifix     = None
        self.src_geotransform     = None
        self.src_projection       = None
        self._patching            = False
        self._cancel_requested    = False

    # -------------------------------------------------------------------------
    # Slot Functions
    # -------------------------------------------------------------------------

    def allChecked(self):
        for cb in [self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh]:
            cb.setChecked(True)
            cb.setEnabled(False)
        self.updateTileCount()

    def customChecked(self):
        for cb in [self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh]:
            cb.setChecked(False)
            cb.setEnabled(True)
        self.updateTileCount()

    def onImageLayerChanged(self, layer):
        if layer:
            self.image_filename    = layer.source()
            self.imageNamePassifix = self.image_filename.rsplit('.', 1)[-1]
            self.lbl_image_info.setText(_image_info(self.image_filename))
        else:
            self.image_filename    = None
            self.imageNamePassifix = None
            self.lbl_image_info.setText('')
        self.updateTileCount()
        self.updateFormatOptions()
        # Refresh mask CRS warning — raster CRS may have changed
        if self.check_mask_enabled.isChecked():
            self.onMaskLayerChanged(self.combo_mask.currentLayer())

    def onMaskLayerChanged(self, layer):
        self.combo_mask_field.clear()
        self.combo_mask_field.addItem('— burn value (binary) —', None)
        if layer and self.check_mask_enabled.isChecked():
            self.mask_layer       = layer
            self.maskNamePassifix = 'tif'   # rasterized output is always GeoTIFF
            for field in layer.fields():
                if field.isNumeric():
                    self.combo_mask_field.addItem(field.name(), field.name())
            info = _vector_mask_info(layer)
            warn = _crs_mismatch(self.combo_input.currentLayer(), layer)
            if warn:
                self.lbl_mask_info.setText(f'{info}\n⚠ {warn}')
                self.lbl_mask_info.setStyleSheet(
                    'font: 8pt "Microsoft JhengHei UI"; color: #b97000; font-style: italic;'
                )
            else:
                self.lbl_mask_info.setText(info)
                self.lbl_mask_info.setStyleSheet(
                    'font: 8pt "Microsoft JhengHei UI"; color: #2c6fad; font-style: italic;'
                )
        else:
            self.mask_layer       = None
            self.maskNamePassifix = None
            self.lbl_mask_info.setText('Select a vector layer to use as mask / label')
            self.lbl_mask_info.setStyleSheet(
                'font: 8pt "Microsoft JhengHei UI"; color: #666; font-style: italic;'
            )

    def onCancelClicked(self):
        if self._patching:
            self._cancel_requested = True
            self.btn_cancel.setEnabled(False)
            self.progress_label.setText('Cancelling — waiting for in-progress tiles to finish…')
        else:
            self.close()

    def pickSavingFolder(self):
        self.saving_folder_name = QFileDialog.getExistingDirectory(self, "Select a Folder", "")
        if self.saving_folder_name:
            self.lineEdit_export.setText(self.saving_folder_name)

    def updateSplitSum(self):
        total = 0
        for field in (self.lineEdit_train, self.lineEdit_test, self.lineEdit_valid):
            text = field.text().strip()
            if text.isnumeric():
                total += int(text)
        if total == 0:
            self.lbl_split_sum.setText('Leave all three empty to skip splitting.')
            self.lbl_split_sum.setStyleSheet('font: 8pt "Microsoft JhengHei UI"; color: #888;')
        elif total == 100:
            self.lbl_split_sum.setText(f'Sum: {total}% \u2713')
            self.lbl_split_sum.setStyleSheet('font: 8pt "Microsoft JhengHei UI"; color: #2a7a2a; font-weight: bold;')
        else:
            self.lbl_split_sum.setText(f'Sum: {total}%  \u2014  must equal 100%')
            self.lbl_split_sum.setStyleSheet('font: 8pt "Microsoft JhengHei UI"; color: #b94040; font-weight: bold;')

    def updateTileCount(self):
        layer = self.combo_input.currentLayer()
        if layer is None:
            self.lbl_tile_count.setText('')
            return
        try:
            pw = int(self.lineEdit_winx.text())
            ph = int(self.lineEdit_winy.text())
            sx = int(self.lineEdit_stridex.text())
            sy = int(self.lineEdit_stridey.text())
        except ValueError:
            self.lbl_tile_count.setText('')
            return
        w = layer.width()
        h = layer.height()
        if pw < 1 or ph < 1 or sx < 1 or sy < 1 or pw > w or ph > h:
            self.lbl_tile_count.setText('')
            return
        cols    = math.floor((w - pw) / sx) + 1
        rows    = math.floor((h - ph) / sy) + 1
        n_tiles = cols * rows
        checks  = [self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh]
        n_aug   = sum(1 for cb in checks if cb.isChecked())
        if n_aug > 1:
            self.lbl_tile_count.setText(
                f'{cols} \u00d7 {rows} grid  \u2192  {n_tiles:,} tiles'
                f'  \u00d7  {n_aug} aug  =  {n_tiles * n_aug:,} samples'
            )
        elif n_aug == 1:
            self.lbl_tile_count.setText(
                f'{cols} \u00d7 {rows} grid  \u2192  {n_tiles:,} tiles'
            )
        else:
            self.lbl_tile_count.setText('')

    def updateFormatOptions(self):
        """Enable PNG/JPEG only in Array Only mode with ≤ 3 bands; GeoTIFF always available."""
        layer       = self.combo_input.currentLayer()
        array_mode  = self.radio_array.isChecked()
        n_bands     = layer.bandCount() if layer else 999
        raster_ok   = array_mode and n_bands <= 3

        model = self.combo_format.model()
        for i, enabled in enumerate([True, raster_ok, raster_ok]):
            item = model.item(i)
            flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            item.setFlags(flags if enabled else Qt.NoItemFlags)

        # If selected format just became unavailable, fall back to GeoTIFF
        if not raster_ok and self.combo_format.currentIndex() > 0:
            self.combo_format.setCurrentIndex(0)

    def toggleMaskInput(self, checked):
        self.combo_mask.setEnabled(checked)
        self.lbl_burn_field.setEnabled(checked)
        self.combo_mask_field.setEnabled(checked)
        self.lbl_foreground.setEnabled(checked)
        self.spin_foreground.setEnabled(checked)
        if checked:
            self.onMaskLayerChanged(self.combo_mask.currentLayer())
        else:
            self.mask_layer       = None
            self.maskNamePassifix = None
            self.combo_mask_field.clear()
            self.combo_mask_field.addItem('— burn value (binary) —', None)
            self.lbl_mask_info.setText('Select a vector layer to use as mask / label')
            self.lbl_mask_info.setStyleSheet(
                'font: 8pt "Microsoft JhengHei UI"; color: #666; font-style: italic;'
            )

    # -------------------------------------------------------------------------
    # Popup Helpers
    # -------------------------------------------------------------------------

    def check_for_mandatory_fillings(self, field_name):
        msg = QMessageBox()
        msg.setWindowTitle('Incomplete form!')
        msg.setText(f"{field_name} has been left unfilled!")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    def popupIncorrect(self, message):
        msg = QMessageBox()
        msg.setWindowTitle('Incorrect data!')
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    def popupIncorrectValue(self):
        msg = QMessageBox()
        msg.setWindowTitle('Incorrect data insertion!')
        msg.setText(
            "Total percentage for Train, Test, and Validation must equal 100.\n"
            "No individual value can exceed 100.\n"
            "Leave all three empty to skip dataset splitting."
        )
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    def popupIncorrectCharacterInsersion(self, name):
        msg = QMessageBox()
        msg.setWindowTitle('Incorrect data insertion!')
        msg.setText(f"{name} must be a positive integer (≥ 1).")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()

    # -------------------------------------------------------------------------
    # Main Patching Logic
    # -------------------------------------------------------------------------

    def _set_patching_ui(self, active):
        self.btn_start.setEnabled(not active)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText('Cancel' if active else 'Close')

    def patchifying(self):
        self._patching         = True
        self._cancel_requested = False
        self._set_patching_ui(True)

        try:
            self._run_patchifying()
        except Exception as exc:
            self.progress_bar.setValue(0)
            self.progress_label.setText('Error — see details.')
            QMessageBox.critical(
                self, 'Patching failed',
                f'An unexpected error stopped the run:\n\n{exc}'
            )
        finally:
            self._patching = False
            self._set_patching_ui(False)

    def _run_patchifying(self):

        # (1) Validate mandatory fields
        if not self.image_filename:
            self.check_for_mandatory_fillings("Input Image")
            return
        export_folder = self.lineEdit_export.text().strip()
        if not export_folder:
            self.check_for_mandatory_fillings("Export Folder")
            return
        if not os.path.isdir(export_folder):
            self.popupIncorrect(
                f'Export folder does not exist:\n{export_folder}\n\n'
                'Please select a valid directory.'
            )
            return
        self.saving_folder_name = export_folder
        if not self.lineEdit_outname.text():
            self.check_for_mandatory_fillings("Base Name")
            return

        # (2) Validate tile size and stride — must be positive integers
        for field, name in [
            (self.lineEdit_winx,    'Tile Size X'),
            (self.lineEdit_winy,    'Tile Size Y'),
            (self.lineEdit_stridex, 'Stride X'),
            (self.lineEdit_stridey, 'Stride Y'),
        ]:
            if not field.text() or not field.text().isnumeric() or int(field.text()) < 1:
                self.popupIncorrectCharacterInsersion(name)
                return

        # (3) Read raster metadata only — no pixels loaded into memory
        meta = _read_raster_meta(self.image_filename)
        if meta is None:
            self.popupIncorrect("Could not open the selected image file.")
            return
        self.src_geotransform = meta['geotransform']
        self.src_projection   = meta['projection']

        # (4) Image dimensions
        self.Width   = meta['width']
        self.Height  = meta['height']
        self.channel = meta['band_count']

        # (5) Patch size and stride
        self.patchSize_x = int(self.lineEdit_winx.text())
        self.patchSize_y = int(self.lineEdit_winy.text())
        self.patchStep_x = int(self.lineEdit_stridex.text())
        self.patchStep_y = int(self.lineEdit_stridey.text())

        # (6) Collect selected augmentation methods
        self.selected_augment_methods = set()
        checks = [
            self.check_original, self.check_rotate90, self.check_rotate180,
            self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh
        ]
        for idx, cb in enumerate(checks):
            if cb.isChecked():
                self.selected_augment_methods.add(idx)

        self.augment_passifixes = ['o', 'r90', 'r180', 'r270', 'fv', 'fh', 'fvh']

        # (7) Validate Train/Test/Validation percentages
        self.Train_val = 0
        self.Test_val  = 0
        self.Valid_val = 0

        for val_attr, field, name in [
            ('Train_val', self.lineEdit_train, 'Train Percentage'),
            ('Test_val',  self.lineEdit_test,  'Test Percentage'),
            ('Valid_val', self.lineEdit_valid,  'Validation Percentage'),
        ]:
            if field.text():
                if field.text().isnumeric():
                    val = int(field.text())
                    if val > 100:
                        self.popupIncorrectValue()
                        return
                    setattr(self, val_attr, val)
                else:
                    self.popupIncorrectCharacterInsersion(name)
                    return

        total_pct = self.Train_val + self.Test_val + self.Valid_val
        if total_pct != 100 and total_pct != 0:
            self.popupIncorrectValue()
            return

        # (8) Resolve vector mask source — rasterization happens per tile in workers
        mask_enabled     = self.check_mask_enabled.isChecked()
        mask_source      = None
        mask_layer_name  = None
        if mask_enabled:
            if self.mask_layer is None:
                self.check_for_mandatory_fillings("Mask / Label Image")
                return
            source = self.mask_layer.source()
            if '|layername=' in source:
                mask_source, mask_layer_name = source.split('|layername=', 1)
            else:
                mask_source = source

        # (9) Create Total output folder — guard against existing output from a prior run
        self.total_path = os.path.join(self.saving_folder_name, "Total")
        existing = [
            p for p in [self.total_path] + [
                os.path.join(self.saving_folder_name, n)
                for n in ('Train', 'Test', 'Validation')
            ]
            if os.path.exists(p)
        ]
        if existing:
            names = ',  '.join(os.path.basename(p) for p in existing)
            reply = QMessageBox.question(
                self, 'Overwrite existing output?',
                f'The following folders already exist in the export directory:\n\n'
                f'  {names}\n\n'
                'They will be deleted and recreated. Continue?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            for p in existing:
                shutil.rmtree(p)
        os.mkdir(self.total_path)
        if mask_enabled:
            total_img_path  = os.path.join(self.total_path, "images")
            total_mask_path = os.path.join(self.total_path, "masks")
            os.mkdir(total_img_path)
            os.mkdir(total_mask_path)
        else:
            total_img_path  = self.total_path
            total_mask_path = None

        # (10) Validate tile size against image size
        if self.patchSize_y > self.Height or self.patchSize_x > self.Width:
            self.popupIncorrect(
                "Tile size cannot exceed the image size!\n"
                f"Image size: {self.Width} x {self.Height}"
            )
            return

        # (11) Sliding window parameters
        self.list_of_saved_names = []
        self.steps_in_height = math.floor((self.Height - self.patchSize_y) / self.patchStep_y) + 1
        self.steps_in_width  = math.floor((self.Width  - self.patchSize_x) / self.patchStep_x) + 1
        total_tiles = self.steps_in_height * self.steps_in_width

        # Snapshot all values before entering threads — no Qt objects, no shared arrays
        tile_kwargs = dict(
            img_path         = self.image_filename,
            n_bands          = self.channel,
            nodata_value     = meta['nodata'],
            mask_source      = mask_source,
            mask_layer_name  = mask_layer_name,
            mask_enabled     = mask_enabled,
            foreground_value = self.spin_foreground.value(),
            burn_field       = self.combo_mask_field.currentData(),
            burn_value       = self.spin_foreground.value(),
            patchStep_y      = self.patchStep_y,
            patchStep_x      = self.patchStep_x,
            patchSize_y      = self.patchSize_y,
            patchSize_x      = self.patchSize_x,
            selected_methods = list(self.selected_augment_methods),
            aug_suffixes     = self.augment_passifixes,
            outname          = self.lineEdit_outname.text(),
            img_ext          = self.combo_format.currentData(),
            mask_ext         = self.maskNamePassifix,
            total_img_path   = total_img_path,
            total_mask_path  = total_mask_path,
            steps_in_width   = self.steps_in_width,
            georef           = self.radio_georef.isChecked(),
            src_geotransform = self.src_geotransform,
            src_projection   = self.src_projection,
        )

        # (12) Crop + augment + save  (single- or multi-threaded)
        n_workers   = self.spin_workers.value()
        tile_errors = []   # (row, col, error_message)

        if n_workers == 1:
            for row in range(self.steps_in_height):
                for col in range(self.steps_in_width):
                    if self._cancel_requested:
                        break
                    try:
                        names = _process_tile(row, col, **tile_kwargs)
                        self.list_of_saved_names.extend(names)
                    except Exception as exc:
                        tile_errors.append((row, col, str(exc)))
                    done = row * self.steps_in_width + col + 1
                    pct  = math.floor(done / total_tiles * 100)
                    self.progress_bar.setValue(pct)
                    self.progress_label.setText(
                        f'Patching: {pct}%  ({done}/{total_tiles} tiles)'
                    )
                    QApplication.processEvents()
                if self._cancel_requested:
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_process_tile, row, col, **tile_kwargs): (row, col)
                    for row in range(self.steps_in_height)
                    for col in range(self.steps_in_width)
                }
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    if self._cancel_requested:
                        for f in futures:
                            f.cancel()
                        break
                    row, col = futures[future]
                    try:
                        self.list_of_saved_names.extend(future.result())
                    except Exception as exc:
                        tile_errors.append((row, col, str(exc)))
                    done_count += 1
                    pct = math.floor(done_count / total_tiles * 100)
                    self.progress_bar.setValue(pct)
                    self.progress_label.setText(
                        f'Patching: {pct}%  ({done_count}/{total_tiles} tiles)'
                    )
                    QApplication.processEvents()

        if tile_errors:
            lines = '\n'.join(
                f'  tile {r * self.steps_in_width + c + 1} (row {r}, col {c}): {msg}'
                for r, c, msg in tile_errors[:10]
            )
            if len(tile_errors) > 10:
                lines += f'\n  … and {len(tile_errors) - 10} more'
            QMessageBox.warning(
                self, f'{len(tile_errors)} tile(s) failed',
                f'{len(tile_errors)} tile(s) could not be processed and were skipped:\n\n{lines}'
            )

        if self._cancel_requested:
            self.progress_bar.setValue(0)
            self.progress_label.setText(
                f'Cancelled — {len(self.list_of_saved_names):,} tiles saved before stopping.'
            )
            return

        self.progress_bar.setValue(100)
        self.progress_label.setText('Patching complete! Starting dataset split...')
        QApplication.processEvents()

        # (13) Dataset splitting
        if total_pct == 100:
            self._split_dataset(mask_enabled, total_img_path, total_mask_path)

        self.progress_bar.setValue(0)
        self.progress_label.setText('All done!')
        QApplication.processEvents()

    def _split_dataset(self, mask_enabled, total_img_path, total_mask_path):

        randomized   = self.list_of_saved_names.copy()
        random.shuffle(randomized)
        total_number = len(randomized)

        if self.Train_val == 0:
            num_train = 0
            if self.Test_val == 0:
                num_test, num_valid = 0, total_number
            elif self.Valid_val == 0:
                num_test, num_valid = total_number, 0
            else:
                num_test  = math.floor((self.Test_val / 100) * total_number)
                num_valid = total_number - num_test
        elif self.Test_val == 0:
            num_test = 0
            if self.Valid_val == 0:
                num_train, num_valid = total_number, 0
            else:
                num_train = math.floor((self.Train_val / 100) * total_number)
                num_valid = total_number - num_train
        elif self.Valid_val == 0:
            num_valid = 0
            num_train = math.floor((self.Train_val / 100) * total_number)
            num_test  = total_number - num_train
        else:
            num_train    = math.floor((self.Train_val / 100) * total_number)
            what_is_left = total_number - num_train
            num_test     = math.floor((self.Test_val / (self.Test_val + self.Valid_val)) * what_is_left)
            num_valid    = what_is_left - num_test

        splits           = [("Train", num_train), ("Test", num_test), ("Validation", num_valid)]
        split_img_paths  = {}
        split_mask_paths = {}

        for split_name, count in splits:
            if count > 0:
                split_root = os.path.join(self.saving_folder_name, split_name)
                os.mkdir(split_root)
                if mask_enabled:
                    img_dir  = os.path.join(split_root, "images")
                    mask_dir = os.path.join(split_root, "masks")
                    os.mkdir(img_dir)
                    os.mkdir(mask_dir)
                    split_img_paths[split_name]  = img_dir
                    split_mask_paths[split_name] = mask_dir
                else:
                    split_img_paths[split_name] = split_root

        saved_count = 0
        for split_name, count in splits:
            if count == 0:
                continue
            img_dest  = split_img_paths[split_name]
            mask_dest = split_mask_paths.get(split_name)

            for _ in range(count):
                img_name = randomized.pop(0)
                shutil.copy2(os.path.join(total_img_path, img_name), os.path.join(img_dest, img_name))
                if mask_enabled:
                    mask_name = img_name.rsplit(".", 1)[0] + "." + self.maskNamePassifix
                    shutil.copy2(os.path.join(total_mask_path, mask_name), os.path.join(mask_dest, mask_name))
                saved_count += 1
                pct = math.floor(saved_count / total_number * 100)
                self.progress_bar.setValue(pct)
                self.progress_label.setText(f'Dataset split: {pct}% completed')
                QApplication.processEvents()
