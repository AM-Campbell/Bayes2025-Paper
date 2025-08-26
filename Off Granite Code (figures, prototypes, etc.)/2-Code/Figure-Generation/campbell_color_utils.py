
from functools import partial


def okabe_ito_palette(format='hex'):
    """Returns the Okabe-Ito color palette in the specified format.

    Args:
        format (str, optional): 'hex', 'rgb', or 'rgba'. Defaults to 'hex'.

    Raises:
        ValueError: If an invalid format is provided. 

    Returns:
        List: List of colors in the specified format. Order: [orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple, gray] 
    """

    if format == 'hex':
        return [
            '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'
        ]
    elif format == 'rgb':
        return [
            (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167), (153, 153, 153)
        ]
    elif format == 'rgba':
        return [
            (230, 159, 0, 1), (86, 180, 233, 1), (0, 158, 115, 1), (240, 228, 66, 1), (0, 114, 178, 1), (213, 94, 0, 1), (204, 121, 167, 1), (153, 153, 153, 1)
        ]
    else:
        raise ValueError(f"Invalid format: {format}")
    
def okabe_ito_palette_dict(format='hex'):
    """Returns the Okabe-Ito color palette as a dictionary.

    Returns:
        Dict: Dictionary of colors in the format {color_name: color_value}
    """

    pallate = okabe_ito_palette(format)

    return {
        'orange': pallate[0],
        'sky blue': pallate[1],
        'bluish green': pallate[2],
        'yellow': pallate[3],
        'blue': pallate[4],
        'vermillion': pallate[5],
        'reddish purple': pallate[6],
        'gray': pallate[7]
    }

def get_longitudinal_care_delivery_paper_colormap():
    colors = okabe_ito_palette_dict('hex')
    output = {
        'diffSignal': colors['blue'],
        'pmpcd': colors['reddish purple'],
        'ci': colors['vermillion'],
        'wmpcd': colors['bluish green'],
        'dbdfe': colors['sky blue'],
        'pcd': colors['orange'],
        'fe': colors['yellow'],
        'gray': colors['gray']
    }

    # convert to rgba string values for css
    def get_alpha_function(alpha=0.2, color=None):
        def get_rbga_string(alpha=0.2):
            return f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"
        return get_rbga_string

    alpha_functions = {}
    for key, value in output.items():
        alpha_functions[key + "_a"] = get_alpha_function(color=value) 
    
    output.update(alpha_functions)
    print(output)
    return output
    

