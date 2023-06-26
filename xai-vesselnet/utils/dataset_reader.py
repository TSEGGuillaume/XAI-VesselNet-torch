def parse_dataset_csv(csv_path, delimiter=';'):
    import csv
    
    x = []
    y = []

    with open(csv_path, mode='r') as csv_file:
        dataset_reader = csv.reader(csv_file, delimiter=delimiter)

        for line in dataset_reader:
            x.append(line[0])
            y.append(line[1])

    return x, y