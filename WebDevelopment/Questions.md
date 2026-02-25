# HTML
1. Which of these statements is correct for defining a table in HTML?
   - [ ] `<table><row><col>Content</col></row></table>`
   - [ ] `<tab><tr><td>Content</td></tr></tab>`
   - [ ] `<table><div><span>Content</span></div></table>`
   - [x] `<table><tr><td>Content</td></tr></table>`

# CSS
2. What does the z-index property do in CSS?
   - [ ] Determines the transparency of an element.
   - [x] Sets the stack level of a positioned element.
   - [ ] Defines the maximum width of an element.
   - [ ] Specifies the border of an element.

# JavaScript
3. What is the output of the following JavaScript code?
   ```javascript
   console.log(0 == false);
   console.log(0 === false);
   ```
   - [ ] true, true
   - [ ] false, true
   - [x] true, false
   - [ ] false, false

4. Which array method in JavaScript returns a new array without modifying the original?
   - [ ] push()
   - [ ] splice()
   - [x] map()
   - [ ] sort()

# React
5. What is the correct way to pass a non-boolean prop to a component in React?
   - [x] `<Component prop="value" />`
   - [ ] `<Component.prop = "value" />`
   - [ ] `Component.prop("value")`
   - [ ] `prop: <Component value />`

6. What is a "key" and why is it important in a list of elements in React?
   - [x] It makes each element unique to React.
   - [ ] It is a way to set events on elements.
   - [ ] It sets the CSS class of the element.
   - [ ] It is used to change the state of a component.

7. What happens if you try to update the state of a React component directly as in the following example?
   ```javascript
   this.state.name = 'New Name';
   ```
   - [ ] The name will be updated correctly.
   - [ ] The app will crash.
   - [x] React will automatically update the UI.
   - [ ] Nothing will happen and the state will not be updated correctly.

# Tailwind CSS
8. In Tailwind CSS, how do you add space between grid items?
   - [x] `grid-gap-4`
   - [ ] `gap-4`
   - [ ] `spacing-4`
   - [ ] `g-4`

9. How do you set a light blue background color in Tailwind CSS?
   - [x] `bg-light-blue`
   - [ ] `bg-blue-100`
   - [ ] `background-blue`
   - [ ] `blue-background`

# TypeScript
10. What is the main difference between interface and type in TypeScript?
    - [ ] `interface` is used only for classes, while `type` is only for functions.
    - [ ] `type` allows union and intersection of types, while `interface` does not.
    - [ ] There is no difference, they are synonyms.
    - [x] `interface` can be implemented by classes, `type` cannot.

11. What does the unknown type indicate in TypeScript?
    - [ ] It is equivalent to any.
    - [ ] It indicates that a variable can be of any type but requires type checking before use.
    - [ ] It means that a variable is undefined.
    - [x] It is a synonym for null.

12. What does the following function return?
    ```typescript
    function sum(a: number, b: number): number | undefined {
      if (a > 0) {
        return a + b;
      }
    }
    ```
    - [ ] Always returns undefined.
    - [x] Returns the sum of a and b if a > 0, otherwise undefined.
    - [ ] Returns the sum of a and b in all cases.
    - [ ] Returns a runtime type error.
