# Auto-generated script to merge split files
import sys

def merge(base, num_parts):
    with open(base, 'wb') as outfile:
        for i in range(num_parts):
            part = f'{base}.part{i:03d}'
            with open(part, 'rb') as infile:
                outfile.write(infile.read())

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python merge_parts.py <filename> <num_parts>')
        sys.exit(1)
    merge(sys.argv[1], int(sys.argv[2]))
