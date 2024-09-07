import os
import sys
import csv

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: " + sys.argv[0] + "path/to/csv start end")
        sys.exit()
    start = float(sys.argv[2])
    end = float(sys.argv[3])
    points = []
    with open(sys.argv[1]) as f:
        d = csv.DictReader(f)
        found_start = True
        for row in d:
            if not found_start:
                if float(row["t"]) == start:
                    found_start = True
            if found_start and float(row["t"]) < end:
                points.append(float(row["v"]))
        
        average = sum(points) / len(points)
        print(average)