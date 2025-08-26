from playwright.sync_api import sync_playwright

def html_to_pdf(html_file_path, output_pdf):
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        
        # Wait for KaTeX to render
        page.evaluate("""
        () => {
            return new Promise((resolve) => {
                if (document.readyState === 'complete') {
                    resolve();
                } else {
                    window.addEventListener('load', resolve);
                }
            }).then(() => {
                return new Promise((resolve) => {
                    setTimeout(resolve, 1000);  // Wait an additional second for rendering
                });
            });
        }
        """)
        
        # Set PDF options for 8.5x11 inch paper
        page.pdf(path=output_pdf, format="Letter", print_background=True)
        browser.close()

# File paths
html_file_path = "methods_figure.html"
output_pdf_path = "methods_figure.pdf"

# Convert HTML to PDF
html_to_pdf(html_file_path, output_pdf_path)
print(f"PDF generated successfully from {html_file_path} to {output_pdf_path}!")