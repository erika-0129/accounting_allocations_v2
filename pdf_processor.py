"""
This script will deal with PDF files only. PDF files have to be filtered at a higher rate than CSV or Excel files
"""

import pytesseract
import pdfplumber
import google.generativeai as genai
import pandas as pd
import os
import logging
import re
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up generative AI model
genai.configure(api_key=os.environ["GeminiAPI"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Ignores details that could disturb the PDF file viewing
EXCLUSION_PATTERNS = [
    r"^Detail Continued",  # Matches "Detail Continued"
    r"^Account Ending",  # Matches "Account Ending"
    r"^p\.\d+/\d+",  # Matches "p. 4/7" (page numbers)
    r"^\s*$",  # Matches blank lines
    r"^Additional Information on the next page"  # Matches Additional Information on the next page
]

HEADER_PATTERNS = [
    r"^Account Ending",  # Matches account summary lines
    r"^Detail Continued",  # Matches detail headers
    r"^Fees and Interest",  # Matches summary text
]


# Remove lines matching exclusion patterns. Commonly found in-between pages with data to be extracted
def filter_irrelevant_lines(text):
    filtered_lines = []
    for line in text.splitlines():
        if not any(re.match(pattern, line) for pattern in EXCLUSION_PATTERNS):
            filtered_lines.append(line)
    return "\n".join(filtered_lines)


# Extract text from a PDF file and filter out irrelevant pages
def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            all_text = ""
            for page_number, page in enumerate(pdf.pages, start=1):
                # Skip the first two pages, which are usually summaries or notices
                if page_number < 3:
                    logging.info(f"Skipping summary or irrelevant page {page_number}.")
                    continue

                page_text = page.extract_text()
                logging.info(f"Processing Page {page_number}:")
                logging.debug(page_text)

                if page_text and contains_relevant_data(page_text):
                    all_text += page_text + "\n"
                else:
                    logging.info(f"Skipping irrelevant page {page_number}.")

            if not all_text.strip():
                logging.warning("No relevant text found in PDF; falling back to OCR.")
                all_text = ocr_extract_text(file_path)

    except Exception as e:
        logging.error(f"Failed to extract text using pdfplumber: {e}. Falling back to OCR.")
        all_text = ocr_extract_text(file_path)

    return all_text


# Check if text contains transaction data (e.g., date, dollar amounts).
def contains_relevant_data(text):
    date_pattern = re.compile(r"(\d{2}/\d{2}/(?:\d{2}|\d{4}))")  # Dates like 10/06/24
    amount_pattern = re.compile(r"\$\d{1,3}(,\d{3})*(\.\d{2})?")  # Dollar amounts
    return bool(date_pattern.search(text) or amount_pattern.search(text))


def parse_transactions(text):
    transaction_pattern = re.compile(
        r"(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)$"  # Handles Debit and Credit separately
    )

    parsed_data = []
    for line in text.splitlines():
        # Skip lines with summary-related keywords
        if any(keyword in line.lower() for keyword in ["summary", "total", "statement", "balance"]):
            logging.info(f"Skipping potential summary row: {line}")
            continue

        # Match line against the regex
        match = transaction_pattern.match(line)
        if match:
            try:
                # Extract fields from regex groups
                date, vendor, amount = match.groups()
                date = date.strip()
                vendor = vendor.strip()
                amount = float(amount.replace("$", "").replace(",", "").strip())

                # Avoid duplicate transactions
                if not any(
                        row[0] == date and row[1] == vendor and row[2] == amount
                        for row in parsed_data
                ):
                    parsed_data.append([date, vendor, amount])
                    logging.debug(f"Added transaction: {date}, {vendor}, {amount}")
                else:
                    logging.info(f"Duplicate transaction skipped: {date} {vendor} {amount}")
            except ValueError as e:
                logging.warning(f"Error parsing line '{line}': {e}")
        else:
            logging.warning(f"Skipping unparseable line: {line}")

    # Return DataFrame
    if parsed_data:
        return pd.DataFrame(parsed_data, columns=["Date", "Vendor", "Amount"])
    else:
        logging.error("No transactions were successfully parsed.")
        return pd.DataFrame(columns=["Date", "Vendor", "Amount"])


def ocr_extract_text(file_path):
    images = convert_from_path(file_path)
    with ThreadPoolExecutor() as executor:
        ocr_results = list(executor.map(pytesseract.image_to_string, images))
    return "\n".join(ocr_results)


# Clean up parsed transactions DataFrame.
def clean_transactions(df):
    # Drop rows with missing or invalid data
    df.dropna(subset=['Date', 'Vendor', 'Amount'], inplace=True)

    # Remove rows where the 'Vendor' column contains metadata-like text
    metadata_patterns = [
        r"^Account Ending",  # Matches account headers
        r"^Detail Continued",  # Matches "Detail Continued"
    ]
    df = df[~df['Vendor'].str.contains('|'.join(metadata_patterns), na=False)]
    return df


def validate_transactions(df):
    if not {'Date', 'Vendor', 'Amount'}.issubset(df.columns):
        raise ValueError("Parsed DataFrame does not contain required columns.")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.dropna(subset=['Date', 'Vendor', 'Amount'], inplace=True)
    return df


# Process a PDF file to extract transactions
def process_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    text = filter_irrelevant_lines(text)
    df_transactions = parse_transactions(text)

    if df_transactions.empty:
        logging.error("No transactions could be parsed from the PDF.")
        raise ValueError("No transactions found.")

    df_transactions = clean_transactions(df_transactions)
    df_transactions = validate_transactions(df_transactions)
    return df_transactions


def find_transaction_start_row(df):
    for index, row in df.iterrows():
        # Check if the row contains a valid date and amount
        if isinstance(row['Date'], str) and isinstance(row['Amount'], (int, float)):
            # Assume that a valid row with Date and Amount is a transaction row
            return index
    logging.warning("No valid transaction rows found.")
    return None


# Process a PDF statement and extract transactions into a DataFrame.
def process_pdf_statement(file_path):
    try:
        # Extract all text from the PDF, excluding irrelevant pages
        text = extract_text_from_pdf(file_path)
        logging.info("Extracted text from PDF.")

        # Parse transactions from the extracted text
        df_transactions = parse_transactions(text)

        if df_transactions.empty:
            logging.error("No transactions could be parsed from the PDF.")
            raise ValueError("No transactions found.")

        # Clean the parsed transactions
        df_transactions = clean_transactions(df_transactions)

        # Check if columns are missing and assign headers programmatically
        expected_columns = ['Date', 'Vendor', 'Amount']
        if list(df_transactions.columns) != expected_columns:
            logging.warning("Headers not found in parsed data. Assigning default headers.")
            df_transactions.columns = expected_columns

        # Enforce data types and handle missing values
        df_transactions['Amount'] = pd.to_numeric(df_transactions['Amount'], errors='coerce')
        df_transactions.dropna(subset=['Amount'], inplace=True)

        logging.info("PDF transactions processed successfully.")
        return df_transactions

    except Exception as e:
        logging.error(f"Error processing PDF file: {e}")
        raise
