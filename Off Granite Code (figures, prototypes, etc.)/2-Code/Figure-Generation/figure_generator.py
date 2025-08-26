from pathlib import Path
import json
from termcolor import colored
import tomli
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright
import base64
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional
from flask import Flask, Response
import threading
import webbrowser
import importlib
import traceback
from campbell_color_utils import get_longitudinal_care_delivery_paper_colormap
import sys

@dataclass
class PageConfig:
    width: str
    height: str
    margin: str
    orientation: str
    dpi: int
    katex_wait_time: int
    content_width: str
    content_height: str
    background_color: str = "black"
    content_background_color: str = "white"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Dict[str, Any]) -> 'PageConfig':
        required_keys = [
            'width', 'height', 'margin', 'orientation', 'dpi', 'katex_wait_time',
            'content_width', 'content_height', 'background_color', 'content_background_color'
        ]
        for key in required_keys:
            if key not in data and key not in defaults:
                raise ValueError(f"Missing required configuration key: {key}")

        return cls(
            width=data.get('width', defaults['width']),
            height=data.get('height', defaults['height']),
            margin=data.get('margin', defaults['margin']),
            orientation=data.get('orientation', defaults['orientation']),
            dpi=data.get('dpi', defaults['dpi']),
            katex_wait_time=data.get('katex_wait_time', defaults['katex_wait_time']),
            content_width=data.get('content_width', defaults['content_width']),
            content_height=data.get('content_height', defaults['content_height']),
            background_color=data.get('background_color', defaults['background_color']),
            content_background_color=data.get('content_background_color', defaults['content_background_color'])
        )

    def to_css(self) -> str:
        return f"""
        @page {{
            size: {self.width} {self.height};
            margin: {self.margin};
            orientation: {self.orientation};
        }}
    
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Arial, sans-serif;
            font-size: 9pt;
            line-height: 1.3;
            margin: 0;
            padding: 0;
            background-color: {self.background_color};
            width: 100vw;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .container {{
            width: {self.content_width};
            height: {self.content_height};
            display: flex;
            flex-direction: column;
            padding: 2px;
            background-color: {self.content_background_color};
            overflow: hidden;
        }}
        """ 

class FigureGenerator:
    def __init__(self, config_path: str):
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "rb") as f:
            self.config = tomli.load(f)

        if 'output_dir' not in self.config:
            raise KeyError("Missing 'output_dir' in config")
        if 'template_dir' not in self.config:
            raise KeyError("Missing 'template_dir' in config")
        if 'default_page_settings' not in self.config:
            raise KeyError("Missing 'default_page_settings' in config")
        if 'figures' not in self.config:
            raise KeyError("Missing 'figures' in config")

        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)

        template_dir = self.config['template_dir']
        if not Path(template_dir).is_dir():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

        try:
            self.plotting_module = importlib.import_module("plotting_functions")
        except ModuleNotFoundError:
            raise ModuleNotFoundError("plotting_functions module not found. Ensure it's in PYTHONPATH.")

        self.app = Flask(__name__)
        self.register_routes()

    def register_routes(self):
        print(colored('running register_routes', 'green'))

        @self.app.route('/<figure_name>')
        def serve_figure(figure_name):
            print(colored('running serve_figure', 'green'))
            try:
                print(colored('running serve_figure try block', 'green'))
                return self._serve_figure_impl(figure_name)
            except Exception as e:
                print(colored('error in serve_figure', 'red'), e)
                traceback.print_exc(file=sys.stderr)
                return Response(f"Internal server error: {str(e)}", status=500, mimetype='text/plain')

            

    def _serve_figure_impl(self, figure_name: str):
        print(colored('running serve_figure_impl', 'green'))
        if 'figures' not in self.config:
            raise KeyError("No 'figures' section in config")

        if figure_name not in self.config['figures']:
            raise ValueError(f"Figure '{figure_name}' not found in configuration.")

        fig_config = self.config['figures'][figure_name]

        # Validate fig_config keys
        if 'template' not in fig_config:
            raise KeyError(f"Missing 'template' key for figure '{figure_name}'")
        
        # Page config
        page_config = PageConfig.from_dict(
            fig_config,
            self.config['default_page_settings']
        )

        # # Data loading
        # data_file = fig_config.get('data', None)
        # data = {}
        # if data_file is not None:
        #     data_path = None
        #     if 'data_path' in self.config:
        #         data_path = Path(self.config['data_path']) / data_file
        #     else:
        #         # If data_path not in global config, assume current dir
        #         data_path = Path(data_file)

        #     if not data_path.is_file():
        #         raise FileNotFoundError(f"Data file not found at {data_path}")

        #     data = self._load_data(data_path)
        #     if not isinstance(data, dict):
        #         raise ValueError("Data file must contain a JSON object.")

        # Data loading

        data_files = fig_config.get('data', None)
        data = {}

        idx_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        if data_files is not None:
            # Convert single path to list for uniform processing
            if not isinstance(data_files, list):
                data_files = [data_files]

            for idx, data_file in enumerate(data_files):
                data_path = None
                if 'data_path' in self.config:
                    data_path = Path(self.config['data_path']) / data_file
                else:
                    # If data_path not in global config, assume current dir
                    data_path = Path(data_file)

                if not data_path.is_file():
                    raise FileNotFoundError(f"Data file not found at {data_path}")

                file_data = self._load_data(data_path)
                if not isinstance(file_data, dict):
                    raise ValueError(f"Data file {data_file} must contain a JSON object.")

                # Add prefixed keys to main data dictionary
                for key, value in file_data.items():
                    prefixed_key = f"{idx_map[idx]}_{key}"
                    data[prefixed_key] = value

        # Generate subfigures
        subfigures = {}
        subfig_list = fig_config.get('subfigures', [])
        print(colored('subfig_list', 'green'), subfig_list)
        if not isinstance(subfig_list, list):
            raise ValueError(f"subfigures must be a list for figure '{figure_name}'.")

        for subfig_name in subfig_list:
            if 'subfigures' not in self.config:
                raise KeyError("No 'subfigures' section in config, but figure references subfigures.")
            
            if subfig_name not in self.config['subfigures']:
                raise ValueError(f"Subfigure '{subfig_name}' not found in 'subfigures' section of config.")

            subfig_config = self.config['subfigures'][subfig_name]
            if 'function' not in subfig_config:
                raise KeyError(f"Missing 'function' key for subfigure '{subfig_name}'")
            if 'figsize' not in subfig_config:
                raise KeyError(f"Missing 'figsize' key for subfigure '{subfig_name}'. Provide a [width, height] array.")

            func_name = subfig_config['function']
            if not hasattr(self.plotting_module, func_name):
                raise AttributeError(f"Function '{func_name}' not found in plotting_functions module.")

            plot_func = getattr(self.plotting_module, func_name)
            if not callable(plot_func):
                raise TypeError(f"'{func_name}' in subfigure '{subfig_name}' is not callable.")

            # Generate the subfigure
            fig = self._generate_subfigure(subfig_config, data, page_config.dpi)
            subfigures[subfig_name] = self._figure_to_base64(fig, page_config.dpi)
            plt.close(fig)

        # Render template
        template_name = fig_config['template']
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}' not found in template directory.")

        data['colors'] = get_longitudinal_care_delivery_paper_colormap()

        html_content = template.render(
            **data,
            subfigures=subfigures,
            page_style=page_config.to_css()
        )

        return Response(html_content, status=200, mimetype='text/html')

    def preview_figure(self, figure_name: str, port: int = 5080):
        """Open the rendered figure in the default web browser"""
        url = f'http://localhost:{port}/{figure_name}'

        def open_browser():
            webbrowser.open(url)
        
        threading.Timer(1.5, open_browser).start()

        print(f"Previewing figure '{figure_name}' at {url}")
        # Run Flask development server

        print(f"Registered routes: {self.app.url_map}")
        url = f'http://localhost:{port}/{figure_name}'
        print(f"Starting Flask server for figure '{figure_name}' at {url}")
        self.app.run(port=port, debug=False, use_reloader=False)


    def _load_data(self, data_path: Path) -> Dict[str, Any]:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file '{data_path}' does not exist.")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON data from '{data_path}': {str(e)}")
        
        if not isinstance(data, dict):
            raise TypeError(f"Data in '{data_path}' must be a JSON object.")
        return data

    def _generate_subfigure(self, subfig_config: Dict[str, Any], data: Dict[str, Any], dpi: int) -> plt.Figure:
        func_name = subfig_config['function']
        plot_func = getattr(self.plotting_module, func_name)

        if 'figsize' not in subfig_config or not isinstance(subfig_config['figsize'], list) or len(subfig_config['figsize']) != 2:
            raise ValueError(f"Invalid 'figsize' in subfigure config '{subfig_config}'. Must be a list of two floats.")

        figsize = subfig_config['figsize']
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        args = subfig_config.get('args', {})
        if not isinstance(args, dict):
            raise ValueError(f"'args' for subfigure '{func_name}' must be a dict.")

        try:
            result = plot_func(fig, ax, data, **args)
        except Exception as e:
            plt.close(fig)
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Error generating subfigure using '{func_name}': {str(e)}")

        if result is None:
            # Some plotting functions return None; assume fig is updated in-place
            return fig
        
        # Some plotting functions return (fig, ax)
        if isinstance(result, tuple) and len(result) == 2:
            returned_fig, returned_ax = result
            if returned_fig is not fig:
                plt.close(fig)
                raise ValueError(f"Plotting function '{func_name}' returned a different figure object than given.")
            # If returned fig matches the original, it's fine.
            return returned_fig

        # If the result is just fig
        if isinstance(result, plt.Figure):
            return result

        # If result is something else unexpected
        raise TypeError(f"Plotting function '{func_name}' returned unexpected type: {type(result)}")

    def _figure_to_base64(self, fig: plt.Figure, dpi: int) -> str:
        print(colored('running figure to base64', 'green'))
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
            print('Saved figure as PNG to buffer') 
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Failed to save figure as PNG: {str(e)}")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def generate_figure(self, figure_name: str):
        print(colored('running generate_figure', 'green'))
        if figure_name not in self.config['figures']:
            raise ValueError(f"No configuration found for figure '{figure_name}'")

        fig_config = self.config['figures'][figure_name]
        page_config = PageConfig.from_dict(fig_config, self.config['default_page_settings'])

        # data = {}
        # data_file = fig_config.get('data', None)
        # if data_file:
        #     if 'data_path' not in self.config:
        #         raise KeyError("Missing 'data_path' in global config.")
        #     data_path = Path(self.config['data_path']) / data_file
        #     data = self._load_data(data_path)

        # Data loading
        data_files = fig_config.get('data', None)
        data = {}

        idx_map = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        if data_files is not None:
            # Convert single path to list for uniform processing
            if not isinstance(data_files, list):
                data_files = [data_files]

            for idx, data_file in enumerate(data_files):
                data_path = None
                if 'data_path' in self.config:
                    data_path = Path(self.config['data_path']) / data_file
                else:
                    # If data_path not in global config, assume current dir
                    data_path = Path(data_file)

                if not data_path.is_file():
                    raise FileNotFoundError(f"Data file not found at {data_path}")

                file_data = self._load_data(data_path)
                if not isinstance(file_data, dict):
                    raise ValueError(f"Data file {data_file} must contain a JSON object.")

                # Add prefixed keys to main data dictionary
                for key, value in file_data.items():
                    prefixed_key = f"{idx_map[idx]}_{key}"
                    data[prefixed_key] = value

        # Generate subfigures
        subfigures = {}
        subfig_list = fig_config.get('subfigures', [])
        print(colored('subfig_list (in generate_figure)', 'green'), subfig_list)
        if not isinstance(subfig_list, list):
            raise ValueError(f"'subfigures' must be a list for figure '{figure_name}'.")

        for subfig_name in subfig_list:
            if 'subfigures' not in self.config or subfig_name not in self.config['subfigures']:
                raise ValueError(f"Subfigure '{subfig_name}' not defined in config.")
            subfig_config = self.config['subfigures'][subfig_name]
            fig = self._generate_subfigure(subfig_config, data, page_config.dpi)
            subfigures[subfig_name] = self._figure_to_base64(fig, page_config.dpi)
            plt.close(fig)

        if 'template' not in fig_config:
            raise KeyError(f"Missing 'template' key for figure '{figure_name}'")
        template_name = fig_config['template']
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}' not found.")

        data['colors'] = get_longitudinal_care_delivery_paper_colormap()
        html_content = template.render(
            **data,
            subfigures=subfigures,
            page_style=page_config.to_css()
        )

        output_path = self.output_dir / fig_config['output_name']
        self._html_to_pdf(html_content, output_path, page_config)

    def _html_to_pdf(self, html_content: str, output_path: Path, page_config: PageConfig):
        if not isinstance(html_content, str):
            raise TypeError("html_content must be a string.")

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError(f"Failed to launch browser: {str(e)}")

            width_inches = float(page_config.width.replace('in', '').strip())
            height_inches = float(page_config.height.replace('in', '').strip())
            width_px = int(width_inches * 96)
            height_px = int(height_inches * 96)

            try:
                context = browser.new_context(viewport={'width': width_px, 'height': height_px})
                page = context.new_page()
                page.set_content(html_content)
                
                # Wait for page to load and then KaTeX if any
                page.evaluate(f"""
                () => new Promise((resolve) => {{
                    if (document.readyState === 'complete') {{
                        setTimeout(resolve, {page_config.katex_wait_time});
                    }} else {{
                        window.addEventListener('load', () => {{
                            setTimeout(resolve, {page_config.katex_wait_time});
                        }});
                    }}
                }})
                """)
                
                pdf_options = {
                    'path': str(output_path),
                    'print_background': True,
                    'width': f"{width_inches}in",
                    'height': f"{height_inches}in",
                    'margin': {
                        'top': "0",
                        'right': "0",
                        'bottom': "0",
                        'left': "0"
                    }
                }

                if page_config.orientation == 'landscape':
                    # Swap width and height for landscape
                    pdf_options['width'], pdf_options['height'] = pdf_options['height'], pdf_options['width']

                page.pdf(**pdf_options)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError(f"Failed to render PDF: {str(e)}")
            finally:
                browser.close()

    def generate_all_figures(self):
        if 'figures' not in self.config:
            raise KeyError("No 'figures' section in config")

        for figure_name in self.config['figures']:
            print(f"Generating {figure_name}...")
            try:
                self.generate_figure(figure_name)
                print(f"Generated {figure_name}")
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError(f"Failed to generate figure '{figure_name}': {str(e)}") from e

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figures from templates defensively')
    parser.add_argument('--config', default="2-Code/Figure-Generation/figure_config.toml",
                        help='Path to figure configuration TOML file')
    parser.add_argument('command', choices=['preview', 'generate-figures'],
                        help='Command to execute (preview or generate-figures)')
    parser.add_argument('--figure', help='Figure name to preview (required for preview command)')

    args = parser.parse_args()

    generator = FigureGenerator(args.config)

    if args.command == 'preview':
        if not args.figure:
            parser.error("--figure is required when using preview command")
        generator.preview_figure(args.figure)
    elif args.command == 'generate-figures':
        print("Generating all figures...")
        generator.generate_all_figures()
