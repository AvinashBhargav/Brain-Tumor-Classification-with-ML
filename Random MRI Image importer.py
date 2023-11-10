from google.colab import files
uploaded = files.upload()
uploaded_filename = list(uploaded.keys())[0]
img = cv2.imread(uploaded_filename, 0)

img_resized = cv2.resize(img, (200, 200))
img_normalized = img_resized.reshape(1, -1) / 255
prediction = sv.predict(img_normalized)

predicted_class = dec[prediction[0]]
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.axis('off')
plt.show()
