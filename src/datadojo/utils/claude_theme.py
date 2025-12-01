from rich.theme import Theme

claude_theme = Theme({
    "info": "#FFCC80",           # Light orange/peach for info messages  
    "warning": "#FFB366",        # Warm orange for warnings
    "danger": "bold #FF6B6B",    # Coral red for errors
    "success": "bold #66BB6A",   # Green for success messages
    "title": "bold #FF9900",     # Claude signature orange for titles
    "header": "bold #FFAD33",    # Golden orange for headers  
    "prompt": "#FFD699",         # Soft peach for prompts
    "command": "bold #FFFFFF",   # White for command text contrast
    "argument": "#FFE0B2",       # Light cream orange for arguments
    "code": "#FFFAF0",          # Floral white for code blocks
    "border": "#CC7A00",        # Darker orange for UI borders
})
