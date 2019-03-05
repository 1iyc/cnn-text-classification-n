# cnn-text-classification-n
refer http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

## train.py

### Excute parameter

```
--data_file=[data file path]
--class_file=[class file path]
--evaluate_every=20000
--checkpoint_every=20000
--dev_sample_percentage=0.000005
--dropout_keep_prob=0.85
--num_epochs=20
--batch_size=256
```

# tensorboard
실행
```
tensorboard --logdir=[checkpointdir]
```
접속
```html
localhost:6006
```
