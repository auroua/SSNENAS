def convert_arch_to_seq(matrix, ops):
    seq = []
    n = len(matrix)
    assert n == len(ops)
    for col in range(1, n):
        for row in range(col):
            seq.append(matrix[row][col]+1)
        if ops[col] == CONV1X1:
            seq.append(3)
        elif ops[col] == CONV3X3:
            seq.append(4)
        elif ops[col] == MAXPOOL3X3:
            seq.append(5)
        if ops[col] == OUTPUT:
            seq.append(6)
    assert len(seq) == (n+2)*(n-1)/2
    return seq


INPUT = 'input'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'