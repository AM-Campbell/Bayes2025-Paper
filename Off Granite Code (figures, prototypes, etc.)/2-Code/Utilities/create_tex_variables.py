import json
import sys
import re

def number_to_words(n):
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
             "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    if n < 20:
        return units[n]
    elif n < 100:
        return tens[n//10] + ("" if n%10 == 0 else units[n%10])
    else:
        return str(n)

def convert_numbers_in_key(key):
    # Find all numbers in the key
    numbers = re.findall(r'\d+', key)
    converted_key = key
    
    # Replace each number with its word equivalent
    for num in sorted(numbers, key=len, reverse=True):
        converted_key = converted_key.replace(num, number_to_words(int(num)))
    
    return converted_key

def convert_json_to_latex(json_data):
    latex_output = []
    
    for key, value in json_data.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
            
        # Convert numbers in key to words before removing underscores
        latex_name = convert_numbers_in_key(key)
        latex_name = latex_name.replace('_', '')
        
        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        latex_command = f"\\newcommand{{\\{latex_name}}}{{{formatted_value}}}"
        latex_output.append(latex_command)
    
    return '\n'.join(latex_output)

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            json_data = json.load(f)
    else:
        json_data = json.loads(sys.stdin.read())
    
    latex_output = convert_json_to_latex(json_data)
    print(latex_output)

if __name__ == "__main__":
    main()