# cnn-text-classification-n
refer http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

## train.py
Train model that classifies words or sentences 

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

## eval.py
Evaluate model using file.
Adding more features....

# tensorboard
Excute
```
tensorboard --logdir=[checkpointdir]
```
Access
```html
localhost:6006
```
