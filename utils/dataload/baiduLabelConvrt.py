def lbcnvrt(label):
    # 1
    label[label == 200] = 1
    label[label == 204] = 1
    label[label == 213] = 1
    label[label == 209] = 1
    label[label == 206] = 1
    label[label == 207] = 1
    # 2
    label[label == 201] = 2
    label[label == 203] = 2
    label[label == 211] = 2
    label[label == 208] = 2
    # 3
    label[label == 216] = 3
    label[label == 217] = 3
    label[label == 215] = 3
    # 4
    label[label == 218] = 4
    label[label == 219] = 4
    # 5
    label[label == 210] = 5
    label[label == 232] = 5
    # 6
    label[label == 214] = 6
    # 7
    label[label == 202] = 7
    label[label == 220] = 7
    label[label == 221] = 7
    label[label == 222] = 7
    label[label == 231] = 7
    label[label == 224] = 7
    label[label == 225] = 7
    label[label == 226] = 7
    label[label == 230] = 7
    label[label == 228] = 7
    label[label == 229] = 7
    label[label == 233] = 7
    # 8
    label[label == 205] = 8
    label[label == 212] = 8
    label[label == 227] = 8
    label[label == 223] = 8
    label[label == 250] = 8
    # 9
    label[label == 249] = 0
    label[label == 255] = 0

    return label


