import random
import string

def key_typos(text: str, probability: float = 0.1) -> str:
    """
    Randomly replaces characters with random alphabet letters to simulate typos.
    """
    if not text: 
        return text
    
    # Get all possible letters (a-z and A-Z)
    alphabet = string.ascii_letters 
    
    text_list = list(text)
    
    for i, char in enumerate(text_list):
        # We check char.isalpha() so we don't replace spaces or punctuation
        if char.isalpha() and random.random() < probability:
            text_list[i] = random.choice(alphabet)
            
    return "".join(text_list)


def keyboard_typos(text: str, probability: float = 0.1) -> str:
        """
        Introduces typos based on keyboard proximity.
        """
        keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr', 
            'f': 'rtgdvc', 'g': 'tyfhbv', 'h': 'yugjbn', 'i': 'ujko', 'j': 'uikhnm',
            'k': 'ijolm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedz', 
            't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 
            'w': 'qase', 'x': 'zsdc', 
            'y': 'tugh', 
            'z': 'asx'
        }
        
        if not text: return text
        
        text_list = list(text)
        for i in range(len(text_list)):
            char = text_list[i].lower()
            if char in keyboard_neighbors and random.random() < probability:
                neighbors = keyboard_neighbors[char]
                typo_char = random.choice(neighbors)
                # Preserve original case
                if text_list[i].isupper():
                    typo_char = typo_char.upper()
                text_list[i] = typo_char
        return "".join(text_list)    

def letter_swaps(text: str, probability: float = 0.1) -> str:
        """
        Randomly swaps adjacent letters in the text.
        """
        if not text: return text
        
        text_list = list(text)
        i = 0
        while i < len(text_list) - 1:
            if random.random() < probability:
                text_list[i], text_list[i+1] = text_list[i+1], text_list[i]
                i += 2  # Skip next character to avoid double swapping
            else:
                i += 1
        return "".join(text_list)



# def make_synonyms(self, text: str, synonym_dict: Optional[dict] = None) -> str:
     

# def make_paraphrase(self, text: str) -> str:


def apply_safe_perturbation(text: str, perturbation_func) -> str:
        """
        Splits text by curly brace placeholders (e.g., {PROMPT_OPTIONS}).
        Applies the perturbation_func ONLY to the text segments, protecting the keys.
        """
        import re
        if not text:
            return text

        parts = re.split(r'(\{.*?\})', text)
        
        processed_parts = []
        for part in parts:
            # Check if this part is a placeholder
            if part.startswith("{") and part.endswith("}"):
                # SAFE ZONE: Append exactly as is
                processed_parts.append(part)
            else:
                # PERTURB ZONE: Apply the typo function
                processed_parts.append(perturbation_func(part))
                
        return "".join(processed_parts)