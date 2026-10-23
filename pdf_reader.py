
from pypdf import PdfReader

try:
    reader = PdfReader("sample.pdf", strict=False)

    print("Total pages:", len(reader.pages))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        print(f"\n--- Page {page_number} ---")

        if text:
            print(text)
        else:
            print("No readable text found on this page.")

except Exception as e:
    print("Error reading PDF:", e)

