import numpy as np
from osgeo import gdal, gdal_array
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QDialog, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QToolButton, QVBoxLayout
)
from PyQt5.QtGui import QIcon
import math
import os
import shutil
import random
from .augment import augment


_EXT_TO_DRIVER = {'tif': 'GTiff', 'tiff': 'GTiff', 'png': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'}


def _read_image(filepath):
    ds = gdal.Open(filepath)
    if ds is None:
        return None
    bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)]
    return bands[0] if ds.RasterCount == 1 else np.stack(bands, axis=-1)


def _save_patch(array, filepath, ext):
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
    ds.FlushCache()
    ds = None


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
        self.setMinimumWidth(660)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # --- Title ---
        title_label = QLabel('Patchify App')
        title_label.setStyleSheet('font: 16pt "Segoe Print";')
        root.addWidget(title_label)

        # --- I/O section ---
        lbl_style = 'font: 10pt "Microsoft JhengHei UI";'

        io_layout = QGridLayout()
        io_layout.setColumnStretch(1, 1)

        label_input = QLabel('Input Image')
        label_input.setStyleSheet(lbl_style)
        self.lineEdit_input = QLineEdit()
        self.btn_input = QToolButton()
        self.btn_input.setText('...')

        label_export = QLabel('Export Folder')
        label_export.setStyleSheet(lbl_style)
        self.lineEdit_export = QLineEdit()
        self.btn_export = QToolButton()
        self.btn_export.setText('...')

        # Mask row — checkbox acts as label; inputs disabled until checked
        self.check_mask_enabled = QCheckBox('Mask / Label Image  (binary & multi-class supported)')
        self.check_mask_enabled.setStyleSheet(lbl_style)
        self.lineEdit_mask = QLineEdit()
        self.lineEdit_mask.setEnabled(False)
        self.btn_mask = QToolButton()
        self.btn_mask.setText('...')
        self.btn_mask.setEnabled(False)

        io_layout.addWidget(label_input,             0, 0)
        io_layout.addWidget(self.lineEdit_input,     0, 1)
        io_layout.addWidget(self.btn_input,          0, 2)
        io_layout.addWidget(label_export,            1, 0)
        io_layout.addWidget(self.lineEdit_export,    1, 1)
        io_layout.addWidget(self.btn_export,         1, 2)
        io_layout.addWidget(self.check_mask_enabled, 2, 0)
        io_layout.addWidget(self.lineEdit_mask,      2, 1)
        io_layout.addWidget(self.btn_mask,           2, 2)
        root.addLayout(io_layout)

        # --- Middle row: Cropping Parameters + Output Options ---
        mid_row = QHBoxLayout()

        gb_crop = QGroupBox('Cropping Parameters:')
        crop_layout = QGridLayout(gb_crop)
        self.lineEdit_winx = QLineEdit()
        self.lineEdit_winx.setMaximumWidth(55)
        self.lineEdit_winy = QLineEdit()
        self.lineEdit_winy.setMaximumWidth(55)
        self.lineEdit_stridex = QLineEdit()
        self.lineEdit_stridex.setMaximumWidth(55)
        self.lineEdit_stridey = QLineEdit()
        self.lineEdit_stridey.setMaximumWidth(55)
        crop_layout.addWidget(QLabel('Window Size:'), 0, 0)
        crop_layout.addWidget(QLabel('X'),            0, 1)
        crop_layout.addWidget(self.lineEdit_winx,     0, 2)
        crop_layout.addWidget(QLabel('Y'),            0, 3)
        crop_layout.addWidget(self.lineEdit_winy,     0, 4)
        crop_layout.addWidget(QLabel('Stride:'),      1, 0)
        crop_layout.addWidget(QLabel('X'),            1, 1)
        crop_layout.addWidget(self.lineEdit_stridex,  1, 2)
        crop_layout.addWidget(QLabel('Y'),            1, 3)
        crop_layout.addWidget(self.lineEdit_stridey,  1, 4)
        mid_row.addWidget(gb_crop)

        gb_output = QGroupBox('Output Options:')
        out_layout = QGridLayout(gb_output)
        out_layout.setColumnStretch(1, 1)
        self.lineEdit_outname = QLineEdit()
        self.lineEdit_train   = QLineEdit()
        self.lineEdit_train.setMaximumWidth(60)
        self.lineEdit_test    = QLineEdit()
        self.lineEdit_test.setMaximumWidth(60)
        self.lineEdit_valid   = QLineEdit()
        self.lineEdit_valid.setMaximumWidth(60)
        out_layout.addWidget(QLabel('Output Name:'),          0, 0)
        out_layout.addWidget(self.lineEdit_outname,           0, 1)
        out_layout.addWidget(QLabel('Training Percentage'),   1, 0)
        out_layout.addWidget(self.lineEdit_train,             1, 1)
        out_layout.addWidget(QLabel('Testing Percentage'),    2, 0)
        out_layout.addWidget(self.lineEdit_test,              2, 1)
        out_layout.addWidget(QLabel('Validation Percentage'), 3, 0)
        out_layout.addWidget(self.lineEdit_valid,             3, 1)
        mid_row.addWidget(gb_output)
        root.addLayout(mid_row)

        # --- Augmentation Options ---
        gb_aug = QGroupBox('Augmentation Options:')
        aug_layout = QGridLayout(gb_aug)
        self.radio_all    = QRadioButton('All')
        self.radio_custom = QRadioButton('Custom Selection')
        self.check_original  = QCheckBox('Original Image')
        self.check_rotate90  = QCheckBox('Rotate 90 degrees')
        self.check_rotate180 = QCheckBox('Rotate 180 degrees')
        self.check_rotate270 = QCheckBox('Rotate 270 degrees')
        self.check_flipv     = QCheckBox('Flip Vertically')
        self.check_fliph     = QCheckBox('Flip Horizontally')
        self.check_flipvh    = QCheckBox('Flip Vertically and Horizontally')
        aug_layout.addWidget(self.radio_all,       0, 0)
        aug_layout.addWidget(self.radio_custom,    1, 0)
        aug_layout.addWidget(self.check_original,  0, 2)
        aug_layout.addWidget(self.check_rotate90,  1, 2)
        aug_layout.addWidget(self.check_rotate180, 2, 2)
        aug_layout.addWidget(self.check_rotate270, 3, 2)
        aug_layout.addWidget(self.check_flipv,     1, 3)
        aug_layout.addWidget(self.check_fliph,     2, 3)
        aug_layout.addWidget(self.check_flipvh,    3, 3)
        root.addWidget(gb_aug)

        # --- Progress label ---
        self.progress_label = QLabel('Patchify is waiting for orders!')
        root.addWidget(self.progress_label)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_start  = QPushButton('Start Patching')
        self.btn_cancel = QPushButton('Cancel')
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        root.addLayout(btn_row)

        # --- Signals & Slots ---
        self.radio_all.toggled.connect(self.allChecked)
        self.radio_custom.toggled.connect(self.customChecked)
        self.btn_cancel.clicked.connect(self.close)
        self.btn_input.clicked.connect(self.pickImage)
        self.btn_export.clicked.connect(self.pickSavingFolder)
        self.btn_mask.clicked.connect(self.pickMaskImage)
        self.check_mask_enabled.toggled.connect(self.toggleMaskInput)
        self.btn_start.clicked.connect(self.patchifying)

        # Default augmentation state
        self.radio_all.setChecked(True)
        self.allChecked()

        # Default values
        self.Train_val           = 0
        self.Test_val            = 0
        self.Valid_val           = 0
        self.list_of_saved_names = []
        self.mask_filename       = None
        self.maskNamePassifix    = None

    # -------------------------------------------------------------------------
    # Slot Functions
    # -------------------------------------------------------------------------

    def allChecked(self):
        for cb in [self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh]:
            cb.setChecked(True)
            cb.setEnabled(False)

    def customChecked(self):
        for cb in [self.check_original, self.check_rotate90, self.check_rotate180,
                   self.check_rotate270, self.check_flipv, self.check_fliph, self.check_flipvh]:
            cb.setChecked(False)
            cb.setEnabled(True)

    def pickImage(self):
        self.image_filename, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "",
            "tif file (*.tif);;png file (*.png);;jpg file (*.jpg)"
        )
        if self.image_filename:
            self.imageNamePassifix = self.image_filename.split(".")[-1]
            self.lineEdit_input.setText(self.image_filename)

    def pickSavingFolder(self):
        self.saving_folder_name = QFileDialog.getExistingDirectory(self, "Select a Folder", "")
        if self.saving_folder_name:
            self.lineEdit_export.setText(self.saving_folder_name)

    def toggleMaskInput(self, checked):
        self.lineEdit_mask.setEnabled(checked)
        self.btn_mask.setEnabled(checked)
        if not checked:
            self.lineEdit_mask.clear()
            self.mask_filename = None

    def pickMaskImage(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select a Mask / Label Image", "",
            "tif file (*.tif);;png file (*.png);;jpg file (*.jpg)"
        )
        if filename:
            self.mask_filename    = filename
            self.maskNamePassifix = filename.split(".")[-1]
            self.lineEdit_mask.setText(filename)

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

    def patchifying(self):

        # (1) Validate mandatory fields
        if not self.lineEdit_input.text():
            self.check_for_mandatory_fillings("Input Image")
            return
        if not self.lineEdit_export.text():
            self.check_for_mandatory_fillings("Export Folder")
            return
        if not self.lineEdit_outname.text():
            self.check_for_mandatory_fillings("Output Name")
            return

        # (2) Validate window size and stride fields — must be positive integers
        for field, name in [
            (self.lineEdit_winx,    'Window Size X'),
            (self.lineEdit_winy,    'Window Size Y'),
            (self.lineEdit_stridex, 'Stride X'),
            (self.lineEdit_stridey, 'Stride Y'),
        ]:
            if not field.text() or not field.text().isnumeric() or int(field.text()) < 1:
                self.popupIncorrectCharacterInsersion(name)
                return

        # (3) Read the image
        self.image = _read_image(self.image_filename)
        if self.image is None:
            self.popupIncorrect("Could not read the selected image file.")
            return

        # (4) Image dimensions
        if self.image.ndim == 2:
            self.Height, self.Width = self.image.shape[:2]
            self.channel = 1
        else:
            self.Height, self.Width, self.channel = self.image.shape[:3]

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

        # (8) Optionally read and validate mask image
        mask_enabled = self.check_mask_enabled.isChecked()
        mask_image   = None
        if mask_enabled:
            if not self.lineEdit_mask.text():
                self.check_for_mandatory_fillings("Mask / Label Image")
                return
            mask_image = _read_image(self.mask_filename)
            if mask_image is None:
                self.popupIncorrect("Could not read the selected mask image file.")
                return
            if mask_image.shape[:2] != (self.Height, self.Width):
                self.popupIncorrect(
                    f"Mask size ({mask_image.shape[1]} x {mask_image.shape[0]}) "
                    f"does not match image size ({self.Width} x {self.Height})."
                )
                return

        # (9) Create Total output folder
        self.total_path = os.path.join(self.saving_folder_name, "Total")
        os.mkdir(self.total_path)
        if mask_enabled:
            total_img_path  = os.path.join(self.total_path, "images")
            total_mask_path = os.path.join(self.total_path, "masks")
            os.mkdir(total_img_path)
            os.mkdir(total_mask_path)
        else:
            total_img_path  = self.total_path
            total_mask_path = None

        # (10) Validate patch size against image size
        if self.patchSize_y > self.Height or self.patchSize_x > self.Width:
            self.popupIncorrect(
                "Patch size cannot exceed the image size!\n"
                f"Image size: {self.Width} x {self.Height}"
            )
            return

        # (11) Reset saved names list for this run
        self.list_of_saved_names = []

        # (12) Sliding window crop + augment + save
        self.steps_in_height = math.floor((self.Height - self.patchSize_y) / self.patchStep_y) + 1
        self.steps_in_width  = math.floor((self.Width  - self.patchSize_x) / self.patchStep_x) + 1

        self.num_of_crop = 0
        for row in range(self.steps_in_height):
            for col in range(self.steps_in_width):
                self.num_of_crop += 1

                y1 = row * self.patchStep_y
                y2 = y1 + self.patchSize_y
                x1 = col * self.patchStep_x
                x2 = x1 + self.patchSize_x

                crop_image = self.image[y1:y2, x1:x2] if self.channel == 1 else self.image[y1:y2, x1:x2, :]
                augment_list = augment(crop_image)

                if mask_enabled:
                    crop_mask = mask_image[y1:y2, x1:x2] if mask_image.ndim == 2 else mask_image[y1:y2, x1:x2, :]
                    mask_augment_list = augment(crop_mask)

                for aug_idx in self.selected_augment_methods:
                    img_name = (
                        f"{self.num_of_crop}_{self.lineEdit_outname.text()}"
                        f"_{self.augment_passifixes[aug_idx]}.{self.imageNamePassifix}"
                    )
                    _save_patch(augment_list[aug_idx], os.path.join(total_img_path, img_name), self.imageNamePassifix)
                    self.list_of_saved_names.append(img_name)

                    if mask_enabled:
                        mask_name = (
                            f"{self.num_of_crop}_{self.lineEdit_outname.text()}"
                            f"_{self.augment_passifixes[aug_idx]}.{self.maskNamePassifix}"
                        )
                        _save_patch(mask_augment_list[aug_idx], os.path.join(total_mask_path, mask_name), self.maskNamePassifix)

                percent = math.floor(self.num_of_crop / (self.steps_in_height * self.steps_in_width) * 100)
                self.progress_label.setText(f'Patching: {percent}% completed')
                QApplication.processEvents()

        self.progress_label.setText('Patching complete! Starting dataset split...')
        QApplication.processEvents()

        # (13) Dataset splitting (only if percentages were provided)
        if total_pct == 100:
            self._split_dataset(mask_enabled, total_img_path, total_mask_path)

        self.progress_label.setText('All done!')
        QApplication.processEvents()

    def _split_dataset(self, mask_enabled, total_img_path, total_mask_path):

        randomized   = self.list_of_saved_names.copy()
        random.shuffle(randomized)
        total_number = len(randomized)

        # Determine counts for each split
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

        # Create split directories
        splits          = [("Train", num_train), ("Test", num_test), ("Validation", num_valid)]
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

        # Copy files
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
                percent = math.floor(saved_count / total_number * 100)
                self.progress_label.setText(f'Dataset split: {percent}% completed')
                QApplication.processEvents()
