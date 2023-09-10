# FocusCam 

---

### FocusCam: The Attention-Guardian Productivity App

---

## Table of Contents

- [About the Project](#about-the-project)
- [Inspiration](#inspiration)
- [Key Learnings](#key-learnings)
- [How We Built It](#how-we-built-it)
- [Dependencies](#dependencies)
- [Setup and Installation](#setup-and-installation)
- [Wrap-Up](#wrap-up)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About the Project

FocusCam uses computer vision and your webcam to monitor your attention levels, ensuring you stay engaged with tasks and minimizing distractions.

---

## Inspiration

The idea was born from our team's struggle with maintaining focus while studying and doing homework. We pondered, "What if our computers could notify us when we're losing attention?" That's how FocusCam was born.

---

## Key Learnings

- **Computer Vision Basics:** Delved into how cameras detect facial nuances using vector math and tracking facial features.
  
- **User Privacy:** Ensured data was processed locally without storing any footage.
  
- **User Interface Design:** Created a user-friendly and non-intrusive interface.

---

## How We Built It

- **Tools:** Utilized OpenCV for computer vision and Electron for the app interface.

- **Detection Algorithm:** Developed to distinguish between reading/thinking and distraction.

- **Real-time Feedback:** Users receive gentle reminders to refocus after a minute of distraction.

- **Privacy:** Data is processed locally; no footage is saved or transmitted.

- **Challenges:** Implementing Gaze Tracking was a significant challenge initially.

---

## Dependencies

The following Python packages are required for FocusCam:

- tkinter
- opencv-python
- numpy
- mediapipe
- pandas
- GazeTracking
- notifypy


---

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tkl0000/FocusCam.git
    ```

2. Navigate to the cloned repository:
    ```bash
    cd FocusCam
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:
    ```bash
    python main.py
    ```

---

## Wrap-Up

FocusCam, from a simple idea to a robust tool, showcases how tech can aid in enhancing daily productivity. Our journey in creating this tool emphasized the importance of persistence and user-centric design.

---

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

---

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) for details.

---

## Contact

For questions, suggestions, or feedback, please reach out to us at thomasli2025@gmail.com.

---

**Enjoy using FocusCam and stay focused!** 📸👀
