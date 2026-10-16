from llm import OpenAIModelProvider


model_provider = OpenAIModelProvider()
model = model_provider.create_model()


def main():
    messages = [
        (
            "system",
            "You are a helpful translator. Translate the user sentence to Portuguese.",
        ),
        ("human", "I love programming."),
    ]
    result = model.invoke(messages)
    print(result)


if __name__ == "__main__":
    main()
