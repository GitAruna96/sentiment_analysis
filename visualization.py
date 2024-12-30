from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate the word cloud from all the cleaned text
text_data = ' '.join(train_data['cleaned_text'])  # Combine all cleaned text into a single string

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud of Cleaned Text')
plt.show()
