indici=[]
for i, value in self.index_mapper.items():
    if value[0] in img_indices:
        indici.append(i)