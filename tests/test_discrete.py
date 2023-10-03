import sspspace


if __name__=='__main__':

    keys = range(10)
    discrete_encoder = sspspace.DiscreteSPSpace(keys, 128)

    v0 = discrete_encoder.encode(0)
    v1 = discrete_encoder.encode(1)

    print(v0|v0)

    print(discrete_encoder.decode(v0))
    print(discrete_encoder.decode(v1))
