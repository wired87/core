from pix2text import Pix2Text

r"""r = process_pdf_bytes(
    pdf_bytes=open(r"C:\Users\bestb\PycharmProjects\BestBrain\test_paper.pdf", "rb")
)
print("r", r)"""


if __name__ =="__main__":
    # Initialisierung (lädt Modelle für Layout, Text und Formeln)
    p2t = Pix2Text.from_config()

    # Ein ganzes PDF analysieren
    pdf_path = r"C:\Users\bestb\PycharmProjects\BestBrain\test_paper.pdf"
    pages = p2t.recognize_pdf(pdf_path)

    for i, page in enumerate(pages):
        # Das Ergebnis ist ein strukturiertes Objekt (Markdown-ähnlich)
        print(f"--- Seite {i + 1} ---")
        print(page.to_markdown('output_dir'))

