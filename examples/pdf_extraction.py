"""
TextIdentifier - PDF Text Extraction Example
Extracts text from PDF files and scanned documents.
"""

import sys
import os
sys.path.insert(0, '..')

from src.documents import DocumentReader

# Folder for test documents (not uploaded to GitHub)
TEST_DOCS_DIR = "../storage/test_docs"


def main():
    print("=" * 60)
    print("PDF & Document Text Extraction")
    print("=" * 60)

    # Create storage folder if it doesn't exist
    if not os.path.exists(TEST_DOCS_DIR):
        os.makedirs(TEST_DOCS_DIR)
        print(f"\nCreated folder: {TEST_DOCS_DIR}/")

    # Get all PDFs and images from storage folder
    extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(TEST_DOCS_DIR) if f.lower().endswith(extensions)]

    if not files:
        print(f"\nNo documents found in {TEST_DOCS_DIR}/")
        print("Add your PDF files or scanned document images there and run again.")
        print("\nSupported formats: PDF, JPG, JPEG, PNG, BMP, TIFF")
        return

    print(f"\nFound {len(files)} document(s) to process.")
    print("Initializing document reader...")

    # Initialize reader
    reader = DocumentReader(languages=["en", "ro"], use_gpu=False)

    for filename in files:
        file_path = os.path.join(TEST_DOCS_DIR, filename)

        print("\n" + "=" * 60)
        print(f"Processing: {filename}")
        print("=" * 60)

        try:
            result = reader.recognize(file_path)

            if filename.lower().endswith('.pdf'):
                # PDF results
                print(f"\n  Pages: {result['total_pages']}")

                if result['metadata']['title']:
                    print(f"  Title: {result['metadata']['title']}")
                if result['metadata']['author']:
                    print(f"  Author: {result['metadata']['author']}")

                print("\n  Text preview (first 500 chars):")
                print("  " + "-" * 50)
                preview = result['full_text'][:500].replace('\n', '\n  ')
                print(f"  {preview}")
                if len(result['full_text']) > 500:
                    print("  ...")
                print("  " + "-" * 50)

                # Word count per page
                print("\n  Pages summary:")
                for page in result['pages']:
                    print(f"    Page {page['page_number']}: {page['word_count']} words" +
                          (" (has images)" if page['has_images'] else ""))

            else:
                # Image results
                print(f"\n  Extracted {result['total_items']} text regions")
                print(f"  Average confidence: {result['avg_confidence']:.1%}")

                print("\n  Text preview:")
                print("  " + "-" * 50)
                preview = result['full_text'][:500]
                print(f"  {preview}")
                if len(result['full_text']) > 500:
                    print("  ...")
                print("  " + "-" * 50)

        except Exception as e:
            print(f"  Error: {e}")


def search_in_pdf():
    """Example of searching text in PDF."""
    print("\n" + "=" * 60)
    print("PDF Search Example")
    print("=" * 60)

    reader = DocumentReader()

    # Find first PDF in storage
    if not os.path.exists(TEST_DOCS_DIR):
        print("No storage folder found.")
        return

    pdfs = [f for f in os.listdir(TEST_DOCS_DIR) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("No PDFs found in storage/")
        return

    pdf_path = os.path.join(TEST_DOCS_DIR, pdfs[0])
    search_term = input("\nEnter search term: ").strip()

    if search_term:
        matches = reader.search_text(pdf_path, search_term)
        print(f"\nFound {len(matches)} matches for '{search_term}':")
        for match in matches[:5]:  # Show first 5
            print(f"  Page {match['page']}: {match['context']}")


if __name__ == "__main__":
    main()
