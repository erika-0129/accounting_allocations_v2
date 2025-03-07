import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
import google.generativeai as genai
import logging
import threading
import pdf_processor
import time
from random import uniform
from datetime import datetime

# Fetch the API key securely from the environment variable
api_key = os.getenv("GeminiAPI")
if not api_key:
    logging.error("GeminiAPI key is missing. Please set the environment variable.")
    raise RuntimeError("Missing GeminiAPI key.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Generate a timestamp
timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

COLUMN_MAP = {
    'debit': 'Amount',
    'credit': 'Amount',
    'due': 'Amount',
    'amount': 'Amount',
    'date': 'Date',
    'transaction date': 'Date',
    'trans date': 'Date',
    'description': 'Vendor',
    'vendor': 'Vendor'

}

# Standardize columns based on the column_map.
# Currently, not in use, but needs to be used at some point since statements do not all have the same column headers
def standardize_columns(df, column_map):
    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Rename columns based on the column map
    df.rename(columns=lambda col: column_map.get(col, col), inplace=True)

    # Apply string transformations (strip and lower) on string columns
    df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == 'object' else col)
    return df


# This function will keep in mind that not all statements have the description column in the first row
def find_header_row(df, column_map):

    """
    required_columns = [col.strip().lower() for col in required_columns]
    for index, row in df.iterrows():
        # Clean up the row values to ignore case and whitespace differences
        row_values = [str(val).strip().lower() for val in row.values]
        if all(column in row_values for column in required_columns):
            return index
    return None
    """
    required_keywords = set(column_map.keys())
    for index, row in df.iterrows():
        row_values = {str(value).strip().lower() for value in row.values if pd.notna(value)}
        logging.info(f"Checking row {index}: {row_values}")
        if required_keywords.intersection(row_values):
            logging.info(f"Header found at row {index}")
            return index
    logging.error("No valid header row found.")
    return None

# Give some pre-defined categories. These can be if they are not easy to find online and can be updated based on
# needs
vendor_categories = {
    'Booking.com': 'Hotel',
    'Delta Air Lines': 'Flight',
    'American Airlines': 'Flight',
    'United Airlines': 'Flight',
    'Marriott': 'Hotel',
    'Godaddy': 'Network',
    'Sunpass': 'Auto Expense',
    'Amazon': 'Shopping'
}

# Checks to make sure entries in the Vendor column are suitable for processing
lower_vendor_categories = {k.lower(): v for k, v in vendor_categories.items()}
def categorize_transaction(vendor):
    if pd.isna(vendor):
        return 'Unknown'
    vendor = vendor.lower()
    return next((category for known_vendor, category in lower_vendor_categories.items() if known_vendor in vendor), 'Unknown')


vendor_cache = {}
# Any vendor that was categorized as unknown will go through the Gemini description generator
def get_vendor_info(vendor):
    if not vendor:
        return " "

    # Data cleaning
    vendor = str(vendor).strip().title()  # Ensure string, strip whitespace, title case

    if vendor in vendor_cache:  # Check cache first
        return vendor_cache[vendor]

    retries = 3  # Set the number of retries
    delay = 2  # Initial delay

    for attempt in range(retries):
        try:
            prompt = (f"Generate a 1 word description of an expense that the following vendor creates {vendor}. "
                      f"If blank, don't return anything")
            response = model.generate_content(prompt)

            if response.text:  # Ensure response is valid
                category = response.text.strip()
                vendor_cache[vendor] = category  # Cache the result
                return category
            else:
                logging.warning(f"Empty response from API for vendor: {vendor}")
                return "Unknown"

        except Exception as e:
            logging.error(f"Error getting AI category for vendor '{vendor}' (attempt {attempt+1}): {e}")
            if "429" in str(e):
                time.sleep(delay)  # Wait and retry with exponential backoff
                delay *= uniform(1.5, 3)
            else:
                return "Unknown"  # Return Unknown for other errors
    logging.error(f"Final API failure for vendor: {vendor}")
    return "Unknown"

def process_statement(file_path, progress, window):
    df = None  # Initialize df to prevent undefined variable errors
    try:
        if file_path.endswith('.pdf'):
            # PDFs don't usually have headers; use preprocessed DataFrame directly
            df = pdf_processor.process_pdf_statement(file_path)
        elif file_path.endswith('.csv') or file_path.endswith('.xlsx'):
            # Handle CSV and Excel files
            read_func = pd.read_csv if file_path.endswith('.csv') else pd.read_excel
            initial_df = read_func(file_path, header=None, nrows=10)
            #required_columns = ['Date', 'Vendor', 'Amount']
            header_row = find_header_row(initial_df, COLUMN_MAP)

            if header_row is not None:
                df = read_func(file_path, header=header_row)
                df = standardize_columns(df, COLUMN_MAP)
            else:
                raise ValueError("Could not find the header row in the uploaded file.")
        else:
            messagebox.showerror("Error", "Unsupported file format. Please upload CSV, Excel, or PDF files.")
            return

        # Standard processing (common to all file types)
        if df is not None:
            df.columns = df.columns.str.strip().str.lower()  # Standardize column names
            #print(f"Columns after standardizing: {df.columns}")
            df = df.rename(columns={col: COLUMN_MAP.get(col, col) for col in df.columns})

            if 'Vendor' not in df.columns:
                messagebox.showerror("Error", "The file does not contain a 'Vendor' column.")
                return

            df.dropna(how='all', inplace=True)
            logging.info(f"DataFrame after dropna: {df.shape}")
            df = df[df['Vendor'].notna()]
            logging.info(f"DataFrame after Vendor notna: {df.shape}")

            logging.info(f"DataFrame before total removal: {df.shape}")
            df = df[~((df['Amount'] == df['Amount'].sum()) & (df.index == df.index[-1]))]
            logging.info(f"DataFrame after total removal: {df.shape}")
            df['Category'] = df['Vendor'].apply(categorize_transaction)

            batch_size = 100  # Adjust batch size as needed
            total_batches = (len(df) + batch_size - 1) // batch_size  # Total number of batches
            progress_step = 100 / total_batches  # Progress increment per batch
            current_progress = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]['Vendor']
                unknown_vendors = batch[batch.isin(df[df['Category'] == 'Unknown']['Vendor'])]

                for vendor in unknown_vendors:
                    category = get_vendor_info(vendor)
                    vendor_mask = df['Vendor'].str.lower() == vendor.lower()
                    df.loc[vendor_mask, 'Category'] = category
                    logging.info("Categorized unknown vendor '%s' as '%s'", vendor, category)

                # Update progress bar after each batch
                current_progress += progress_step / len(unknown_vendors)
                progress["value"] = current_progress
                window.update_idletasks()

            category_summary = df.groupby('Category')['Amount'].sum().reset_index()

            try:
                output_path = os.path.join(os.getcwd(), f'allocated_report_{timestamp}.csv')
                if df is None or df.empty:
                    logging.error("DataFrame is empty. No file written.")
                else:
                    logging.info(f"DataFrame Preview:\n{df.head()}")  # Add this line
                    logging.info(f"DataFrame Shape: {df.shape}")  # added
                    logging.info(f"DataFrame Data Types: {df.dtypes}")  # added
                    print(df.to_string())  # Added
                    logging.info(f"Current working directory: {os.getcwd()}")
                    df.to_csv(output_path, index=False, encoding='utf-8') #added encoding
                    logging.info(f"File saved successfully: {output_path}")
            except Exception as e:
                logging.error(f"Error saving CSV file: {e}")
                messagebox.showerror("Error", f"Failed to save CSV: {e}")

            progress["value"] = 100  # Ensure progress bar is full at completion
            window.update_idletasks()

            messagebox.showinfo("Success", f"Report generated and saved to {output_path}")
        else:
            messagebox.showerror("Error", "Could not find the header row in the uploaded file.")
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        messagebox.showerror("Error", f"Processing failed: {str(e)}")

# Run the processing logic in a separate thread.
def process_statement_in_thread(file_path, progress, window):
    threading.Thread(
        target=process_statement, args=(file_path, progress, window), daemon=True
    ).start()

def upload_file_with_progress(progress, window):
    file_path = filedialog.askopenfilename(
        filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("PDF files", "*.pdf"))
    )
    if file_path:
        progress["value"] = 0  # Reset the progress bar to 0
        # Initially hide the progress bar
        progress.pack_forget()
        # Start the processing in a separate thread
        process_statement_in_thread(file_path, progress, window)
        # Show the progress bar after the thread starts
        window.after(100, lambda: progress.pack(pady=10))

# Setting up the main GUI window
def main_screen():
    window = tk.Tk()
    window.title("Expense Categorizer")
    window.geometry("400x250")

    label = tk.Label(window, text="Upload Your Credit Card Statement.\nExcel, CSV, and PDF files accepted",
                     font=("Helvetica", 12))
    label.pack(pady=20)

    progress = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")

    button_frame = tk.Frame(window)
    button_frame.pack(pady=20)

    upload_button = tk.Button(button_frame, text="Upload File",
                               command=lambda: upload_file_with_progress(progress, window), font=("Helvetica", 10))
    upload_button.pack(side=tk.LEFT, padx=10)

    def close_window():
        window.destroy()

    close_button = tk.Button(button_frame, text="Close", command=close_window, font=("Helvetica", 10))
    close_button.pack(side=tk.LEFT, padx=10)

    window.mainloop()


if __name__ == "__main__":
    main_screen()