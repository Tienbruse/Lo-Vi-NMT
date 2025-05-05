def find_line_with_most_tokens(file_path):
    max_token_count = 0
    max_line = ""
    max_line_number = -1

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            tokens = line.strip().split()
            token_count = len(tokens)
            if token_count > max_token_count:
                max_token_count = token_count
                max_line = line.strip()
                max_line_number = idx

    print(f"Dòng có nhiều token nhất là dòng {max_line_number} với {max_token_count} token:")
    #print(max_line)

# Ví dụ sử dụng:
# find_line_with_most_tokens("path/to/your/file.txt")

if __name__ == "__main__":
    find_line_with_most_tokens('processed_data/dev.lo')
    find_line_with_most_tokens('processed_data/dev.vi')
    find_line_with_most_tokens('processed_data/test.lo')
    find_line_with_most_tokens('processed_data/test.vi')
    find_line_with_most_tokens('processed_data/train.lo')
    find_line_with_most_tokens('processed_data/train.vi')