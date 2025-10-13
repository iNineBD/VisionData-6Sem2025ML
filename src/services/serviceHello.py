def hello():
    """Simple hello function."""
    return "Hello, VisionData!"


def add(a, b):
    """Add two numbers."""
    return a + b


def main():
    """Main function."""
    print(hello())
    result = add(2, 3)
    print(f"2 + 3 = {result}")


if __name__ == "__main__":
    main()
