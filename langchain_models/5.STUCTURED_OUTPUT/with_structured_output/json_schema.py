from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# from typing import Optional, Literal, Annotated
# from pydantic import BaseModel, Field
import json

load_dotenv()

CustomerReview_json_schema = {
  "title": "CustomerReview",
  "type": "object",
  "properties": {
    "summary": {
      "description": "A brief summary of the customer review in 50-70 words",
      "title": "Summary",
      "type": "string"
    },
    "sentiment": {
      "description": "The overall sentiment of the review either positive, negative or neutral based on the content of the review",
      "enum": [
        "positive",
        "negative",
        "neutral"
      ],
      "title": "Sentiment",
      "type": "string"
    },
    "pros": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": None,
      "description": "A list of top 3 positive aspects mentioned in the review if available and explicitely mentioned as Pros:, otherwise empty list",
      "title": "Pros"
    },
    "cons": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": None,
      "description": "A list of top 3 negative aspects mentioned in the review if available and explicitely mentioned as Cons:, otherwise empty list",
      "title": "Cons"
    }
  },
  "required": [
    "summary",
    "sentiment"
  ],
}

# Create an instance of the ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

structured_model = model.with_structured_output(CustomerReview_json_schema)

# Define a prompt for the model to generate a customer support response
review = """ One important point about delivery and exchange experience — I opted for a trade-off with my iPhone 12, and the delivery personnel experience was disappointing. They were quite rude and pressured me by saying the trade-off could get cancelled and I would have to pay the full amount. On top of that, they asked for extra money via personal UPI, which was not appropriate at all. I had to pay them that amount. This part of the experience definitely needs improvement.
After using the Samsung Galaxy Z Fold 7 (Grey) for around 3 months, I can confidently say this is not just a phone — it’s a productivity machine, but with a few trade-offs you should be aware of.

First thing, the design. The grey finish looks premium, subtle, and mature — not flashy, but it gives that “serious user” vibe. The folding mechanism feels refined and solid. Initially I was extra cautious, but over time it has proven to be quite durable for daily use.

Now coming to the real experience — the inner display is the game changer. Whether it’s multitasking, reading, watching content, or even working on documents, it genuinely replaces the need for a tablet. Once you get used to this screen, going back to a normal phone feels restrictive.

Performance-wise, it’s smooth and powerful. I’ve used it for everything from heavy multitasking to social media and media consumption — no lag, no heating issues in my usage pattern.

Battery is decent, not exceptional. It comfortably lasts a day with moderate to heavy use, but considering the size and capability, I expected slightly better optimization.

Camera is good, but not “ultra flagship” level. It gets the job done for social media and regular photography, but if camera is your top priority, there are better options in the same price range.

Pros:
• Massive immersive inner display (true productivity boost)
• Premium and clean grey finish
• Smooth performance with no noticeable lag
• Excellent multitasking capabilities
• Feels like a phone + tablet combo

Cons:
• Battery could be better for power users
• Camera is good, but not top-tier flagship level
• Slightly bulky and takes time to adjust
• Expensive — definitely not for casual users.
This device is for a very specific type of user — someone who values productivity, multitasking, and a unique experience over everything else. If you just want a normal smartphone experience, this might feel like overkill. But if you want to level up how you use your phone daily, this is worth it.

Personally, after 3 months, I don’t think I can go back to a regular phone easily. """

# Generate a response from the model
response = structured_model.invoke(review)

# print(response)

# Accessing the structured output
print("Customer Review Summary:", response["summary"])
print("Sentiment Analysis:", response["sentiment"])
print("Pros:", response["pros"])
print("Cons:", response["cons"])

# schema_obj = CustomerReview.model_json_schema()
# print(json.dumps(schema_obj, indent=2))