# formats.py  – board-to-text renderers
# ------------------------------------

def format_simple(board):
    return '\n'.join(' '.join(row) for row in board)

def format_with_labels(board):
    width = len(board[0])
    column_numbers = '  ' + ' '.join(str(i + 1) for i in range(width))
    rows = []
    for idx, row in enumerate(board):
        row_label = chr(ord('A') + idx)
        rows.append(f"{row_label} " + ' '.join(row))
    return '\n'.join([column_numbers] + rows)

def format_table(board):
    width = len(board[0])
    header = '| ' + ' | '.join(str(i + 1) for i in range(width)) + ' |'
    separator = '|---' * width + '|'
    rows = ['| ' + ' | '.join(row) + ' |' for row in board]
    return '\n'.join([header, separator] + rows)

def format_bordered(board):
    width = len(board[0])
    line = '.' + '---.' * width
    body = ['|' + '|'.join(f' {c} ' for c in row) + '|' for row in board]
    rows = []
    for r in body:
        rows.append(line)
        rows.append(r)
    rows.append(line)
    return '\n'.join(rows)

def format_square_brackets(board):
    return '\n'.join('[' + ']['.join(' ' if c == '.' else c for c in row) + ']' for row in board)

def format_commas(board):
    return '\n'.join(','.join(row) for row in board)

def format_underlined(board):
    width = len(board[0])
    column_numbers = ' ' + ' '.join(str(i + 1) for i in range(width))
    underline = '-' * (2 * width - 1)
    rows = ['|' + '|'.join(' ' if c == '.' else c for c in row) + '|' for row in board]
    return '\n'.join([column_numbers, underline] + rows + [underline])

# List of all renderers – the dataset builder will choose randomly
FORMATTERS = [
    format_simple,
    format_with_labels,
    format_table,
    format_bordered,
    format_square_brackets,
    format_commas,
    format_underlined,
]

