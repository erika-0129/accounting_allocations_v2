## Setting Up OpenAI API Key
To run this project, you'll need an OpenAI API key. Please follow the steps below to securely set it up:

### Windows
- Open "Edit the system environment variables."
- Click on "Environment Variables" and add a new user variable called `OPENAI_API_KEY` with your API key as its value.

### macOS/Linux
- Open the terminal and edit your shell profile (e.g., `~/.bash_profile`, `~/.zshrc`, or `~/.bashrc`).
- Add the following line:
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```
## Using The Program
After API key has loaded, the user should be able to run the program as 
  ```bash 
  python accounting_allocations.py
  ```
This will open a dialog box prompting the user to upload a CSV, Excel or PDF file of what they want allocated.

Usually this would be a credit card statement.

The program will use AI to analyze each expense and give it a description.
It will export a CSV file with a "Detailed Report" tab that returns all line items with its category, and a "Category Summary"
tab that sums up each category to easily see how much the user spent in that category that month.

### accounting_allocations.py
This is the driver program. It contains the functionality to use Gemini AI and creates a dialog box to upload a file.
It also mainly processes CSV and Excel files.

### AccountingPDF.py
Due to the complexity of PDF statements, a separate script was produced.
This will review PDF files and finds the transactions to be allocated.

### pdf_processor.py
This script will extract text from PDF. Uses Optical Character Recognition in case the PDF is non-editable