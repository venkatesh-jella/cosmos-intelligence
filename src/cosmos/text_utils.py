import xml.etree.ElementTree as ET
def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

SUBSCRIPT_MAP = str.maketrans(
    "0123456789",
    "₀₁₂₃₄₅₆₇₈₉"
)

def to_subscript(text: str) -> str:
    return text.translate(SUBSCRIPT_MAP)

def extract_scientific_text(element):
    result = element.text or ""

    for child in element:
        child_text = extract_scientific_text(child)

        if child.tag.endswith("}sub") or child.tag == "sub":
            child_text = to_subscript(child_text)

        result += child_text
        result += child.tail or ""

    return result
if __name__ == "__main__":
    sample = "JWST    detected\n carbon dioxide"
    print(normalize_whitespace(sample))

    print(to_subscript("2"))
    print(to_subscript("12"))

    sample_xml = ET.fromstring(
        "<p>JWST detected CO<sub>2</sub> in the atmosphere.</p>"
    )

    result = extract_scientific_text(sample_xml)
    print("Extracted:", result)