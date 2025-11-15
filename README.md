# quickdraw
a nice wrapper for google's [quickdraw dataset](https://quickdraw.withgoogle.com/data)

```
pip install git+https://github.com/Mayukhdeb/quickdraw.git
```

```python
dataset = QuickDrawDataset(
    split="train",
    image_size=(224, 224),
    cache_dir = "./hf_cache",
    custom_class_names=None
)

sample = dataset[0]

print(f"Sample label: {sample['label']}, name: {sample['name']}")
sample["image"].save("sample.jpg")
```

You can also load a custom set of classes like this, but note that this would change the label IDs

```python
dataset = QuickDrawDataset(
    split="train",
    image_size=(224, 224),
    cache_dir = "./hf_cache",
    custom_class_names=["face", "arm", "beach"],
)
```