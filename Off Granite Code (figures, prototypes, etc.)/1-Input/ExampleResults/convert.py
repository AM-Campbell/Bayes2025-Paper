from playwright.sync_api import sync_playwright
import os

def convert_html_to_pdf(html_path, output_path):
    """
    Convert an HTML file to PDF using Playwright with exact page dimensions.
    
    Args:
        html_path (str): Path to the HTML file
        output_path (str): Path where the PDF should be saved
    """
    abs_path = os.path.abspath(html_path)
    file_url = f'file://{abs_path}'
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Create context with exact viewport size
        context = browser.new_context(
            viewport={'width': 672, 'height': 480}  # 7in and 5in in pixels (96 DPI)
        )
        
        page = context.new_page()
        page.goto(file_url)
        
        # Wait for KaTeX to render
        page.wait_for_timeout(1000)
        
        # PDF options for exact size with no margins
        pdf_options = {
            'width': '7in',
            'height': '4.5in',
            'margin': None,  # This ensures absolutely no margins
            'print_background': True,
            'scale': 1.0
        }
        
        # Generate PDF
        page.pdf(path=output_path, **pdf_options)
        
        browser.close()

if __name__ == "__main__":
    # Example usage
    html_file = "1-Input/ExampleResults/example-results-1.html"  # Replace with your HTML file path
    pdf_file = "3-Output/GeneratedFigures/example_results.pdf"        # Replace with desired output path
    
    try:
        convert_html_to_pdf(html_file, pdf_file)
        print(f"Successfully created PDF: {pdf_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")