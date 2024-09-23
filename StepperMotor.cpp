#define F_CPU 16000000UL // Define the CPU clock speed (16 MHz for AVR)

#include <avr/io.h>
#include <util/delay.h>

#define STEP_PIN PD2 // Define step pin (connected to 2nd digital pin)
#define DIR_PIN PD3 // Define direction pin (connected to 3rd digital pin)
#define BUTTON_PIN PD5 // Define button pin (connected to 5th digital pin)

class StepperMotor {
public:
    StepperMotor() {
        // Set STEP_PIN and DIR_PIN as outputs
        DDRD |= (1 << STEP_PIN) | (1 << DIR_PIN);
        
        // Set BUTTON_PIN as input
        DDRD &= ~(1 << BUTTON_PIN);
        
        // If using internal pull-up resistor (for active-low button configuration)
        // PORTD |= (1 << BUTTON_PIN);
    }

    void step() {
        // Set STEP_PIN high
        PORTD |= (1 << STEP_PIN);
        _delay_us(1000); // Delay for step pulse width
        
        // Set STEP_PIN low
        PORTD &= ~(1 << STEP_PIN);
        _delay_us(1000); // Delay between steps
    }

    void rotate(int steps, int direction) {
        if (direction == 1) {
            // Set DIR_PIN high for one direction
            PORTD |= (1 << DIR_PIN);
        } else {
            // Set DIR_PIN low for the opposite direction
            PORTD &= ~(1 << DIR_PIN);
        }

        for (int i = 0; i < steps; i++) {
            step(); // Perform one step
        }
    }

private:
    // Additional private members can be added here if needed
};

int main() {
    StepperMotor motor; // Create a StepperMotor object

    _delay_ms(500); // Initial delay before starting

    uint8_t button_pressed = 0; // Flag to track button press

    while (1) {
        // Check if BUTTON_PIN is high (button pressed)
        if ((PIND & (1 << BUTTON_PIN)) && !button_pressed) {
            _delay_ms(50);
            if (!(PIND & (1 << BUTTON_PIN))) {
                button_pressed = 1;
                
                // Rotate 180 degrees in one direction (assuming 100 steps equals 180 degrees)
                motor.rotate(100, 1);

                _delay_ms(3000);

                // Rotate back to the original position
                motor.rotate(100, 0);
                
                if (!(PIND & (1 << BUTTON_PIN)) && button_pressed) {
                    button_pressed = 0; // Reset the flag for the next button press
                }
                
                _delay_ms(1000);
            }
        }
    }

    return 0;
}
