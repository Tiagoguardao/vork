"""Original geometric COLRv0 fixtures. Regenerate with uv run --with fonttools generate.py."""
from pathlib import Path

from fontTools.colorLib.builder import buildCOLR, buildCPAL
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def rectangle(left, right, top):
    pen = TTGlyphPen(None)
    if top:
        pen.moveTo((left, 0))
        pen.lineTo((left, top))
        pen.lineTo((right, top))
        pen.lineTo((right, 0))
        pen.closePath()
    return pen.glyph()


for name, advance, right, color in [
    ("Wide", 1000, 800, (1, 0, 0, 1)),
    ("Narrow", 500, 300, (0, 0, 1, 1)),
]:
    font = FontBuilder(1000, isTTF=True)
    glyphs = [".notdef", "symbol", "layer", "indicator", "flag", "tag", "end", "joined", "short"]
    font.setupGlyphOrder(glyphs)
    font.setupCharacterMap({
        0x2705: "symbol", 0x1F7E5: "symbol", 0x1F1FD: "indicator",
        0x1F3F4: "flag", 0xE0078: "tag", 0xE007F: "end",
    })
    ink = {g: rectangle(0, 0, 0) for g in glyphs}
    ink["layer"] = rectangle(100, right, 600)
    ink["short"] = rectangle(100, right, 300)
    font.setupGlyf(ink)
    font.setupHorizontalMetrics({g: (advance, 100 if g in ("layer", "short") else 0) for g in glyphs})
    font.setupHorizontalHeader(ascent=800, descent=-200)
    # Same family and style on purpose. A family-name lookup cannot distinguish these files.
    font.setupNameTable({"familyName": "Colour Probe", "styleName": "Regular", "psName": "ColourProbe" + name})
    font.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    font.setupPost()
    features = """
        feature liga {
            sub indicator indicator by joined;
            sub flag tag end by joined;
        } liga;
        feature kern { pos symbol symbol -125; } kern;
    """
    if name == "Narrow":
        features = features.replace("sub flag tag end by joined;", "")
    addOpenTypeFeaturesFromString(font.font, features)
    font.font["COLR"] = buildCOLR({g: [(layer, 0)] for g, layer in [
        ("symbol", "layer"), ("flag", "layer"), ("joined", "short")
    ]}, version=0)
    font.font["CPAL"] = buildCPAL([[color]])
    font.font["head"].created = font.font["head"].modified = 3800000000
    font.save(Path(__file__).with_name(name + ".ttf"))
