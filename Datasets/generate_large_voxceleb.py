
import os

ids = [d for d in os.listdir(VOX_CELEB_LOCATION) if d[0:2] == 'id']


train = ids[0:int(0.7*len(ids))]
val = ids[int(0.7*len(ids)):int(0.8*len(ids))]
test = ids[int(0.8*len(ids)):]

import numpy as np
np.save('./large_voxceleb/train.npy', np.array(train))
np.save('./large_voxceleb/test.npy', np.array(test))
np.save('./large_voxceleb/val.npy', np.array(val))

print(np.array(val).shape)
print(val[0])
