from .camie_tagger import CamieTaggerNode

NODE_CLASS_MAPPINGS = {
    "CamieTaggerNode": CamieTaggerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerNode": "Camie Tagger"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Camie Tagger: Custom node loaded successfully.")