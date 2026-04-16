def classFactory(iface):
    from .patchify import PatchifyPlugin
    return PatchifyPlugin(iface)
