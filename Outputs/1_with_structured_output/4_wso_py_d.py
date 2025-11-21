from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

'''
Optional ---> means a field may be missing
Literal ---> is used when you want to restrict a value to a very specific set of allowed choices.
'''
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = api_key
)

# schema
class Review(BaseModel):
    # summary: str
    # sentiment: str
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal['pos', 'neg'] = Field(description="Return sentiment of the review either positive, negavtive or neutral")
    key_themes: list[str] = Field(description="Write down all the key themes disscussed in the review in a list")
    pros: list[str] = Field(description="Write down all pros inside a list")
    cons: list[str] = Field(description="Write down all cons inside a list")
    reviewer: Optional[str] = Field(default=None, description="Name of the reviewer, or null if not provided")

'''
- look like magic, LLM automatically gives summary and sentiment but it's not magic!
- langChain compiled TypedDict into a schema and Gemini produced JSON that fits that schema
- the model is not guessing, it is actually receiving a strict schema that tells it exactly what fields to produce
'''

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    '''
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

    Pros:
    Insanely powerful processor (great for gaming and productivity)
    Stunning 200MP camera with incredible zoom capabilities
    Long battery life with fast charging
    S-Pen support is unique and useful

    Cons:
    Bulky and heavy—not great for one-handed use
    Bloatware still exists in One UI
    Expensive compared to competitors
    '''
)

print(result)


