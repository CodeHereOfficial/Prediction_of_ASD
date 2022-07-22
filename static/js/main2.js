(function(){
    function buildQuiz(){
      // variable to store the HTML output
      const output = [];
  
      // for each question...
      myQuestions.forEach(
        (currentQuestion, questionNumber) => {
  
          // variable to store the list of possible answers
          const answers = [];
  
          // and for each available answer...
          for(letter in currentQuestion.answers){
  
            // ...add an HTML radio button
            answers.push(
              `<label>
                <input type="radio" name="question${questionNumber}" value="${letter}">
                ${letter} :
                ${currentQuestion.answers[letter]}
              </label>`
            );
          }
  
          // add this question and its answers to the output
          output.push(
            `<div class="question"> ${currentQuestion.question} </div>
            <div class="answers"> ${answers.join('')} </div>`
          );
        }
      );
  
      // finally combine our output list into one string of HTML and put it on the page
      quizContainer.innerHTML = output.join('');
    }
  
    function showResults(){
  
      // gather answer containers from our quiz
      const answerContainers = quizContainer.querySelectorAll('.answers');
  
      // keep track of user's answers
      let numCorrect = 0;
  
      // for each question...
      myQuestions.forEach( (currentQuestion, questionNumber) => {
  
        // find selected answer
        const answerContainer = answerContainers[questionNumber];
        const selector = `input[name=question${questionNumber}]:checked`;
        const userAnswer = (answerContainer.querySelector(selector) || {}).value;
  
        // if answer is correct and counting it
        if(userAnswer === currentQuestion.correctAnswer){
          // add to the number of correct answers
          numCorrect++;
        }
      });
  
      // show number of correct answers out of total
      resultsContainer.innerHTML = `${numCorrect} out of ${myQuestions.length}`;
    }
  
  
    const quizContainer = document.getElementById('quiz');
    const resultsContainer = document.getElementById('results');
    const submitButton = document.getElementById('submit');
    const myQuestions = [
      {
        question: "Does the adult have repetitive behaviour?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Does the adult face lack in eye contact?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Does the adult non verbal?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "b"
      },
      {
        question: "Is there delay in there speech?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Does the adult have obessive interest?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "b"
      },
      {
        question: "Is there lack of social skills?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Facing difficulty relating to people?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "b"
      },
      {
        question: "Having social anxiety?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Having bad executive function?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "b"
      },
      {
        question: "Having poor communicatin Skills?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Relation with Adult",
        answers: {
          a: "Health care professional",
          b: "others",
          c: "parents",
          d: "relative",
          e: "self"
          
        },
        correctAnswer: "c"
      },
      {
        question: "Age ?",
        answers: {
          a: "18 and more",
          b: "less than 18"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Used app before?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "b"
      },
      {
        question: "Country of Residency?",
        answers: {
          a: "India",
          b: "USA",
          c: "China",
          d: "Japan",
          e: "others"
          
        },
        correctAnswer: "a"
      },
      {
        question: "do you have Jaundice?",
        answers: {
          a: "Yes",
          b: "NO"
          
        },
        correctAnswer: "a"
      },
      {
        question: "Gender",
        answers: {
          a: "male",
          b: "Female"
          
        },
        correctAnswer: "b"
      }

    ];
  
    // Kick things off
    buildQuiz();
  
    // Event listeners
    submitButton.addEventListener('click', showResults);
  })