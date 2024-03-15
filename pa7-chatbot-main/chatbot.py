# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np
import re
import random

from porter_stemmer import PorterStemmer

class EmotionDetector(BaseModel):
    Anger: bool = Field(default=False)
    Disgust: bool = Field(default=False)
    Fear: bool = Field(default=False)
    Happiness: bool = Field(default=False)
    Sadness: bool = Field(default=False)
    Surprise: bool = Field(default=False)

class InputCategorizer(BaseModel):
    is_movie_related: bool = Field(default=True)
    is_general_inquiry: bool = Field(default=False)
    is_assistance_request: bool = Field(default=False)
    is_off_topic: bool = Field(default=False)

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'
        self.stemmer = PorterStemmer()
        self.llm_enabled = llm_enabled
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        #self.binarized_ratings = self.binarize(self.ratings, threshold=2.5)
        self.ratings = self.binarize(self.ratings, threshold=2.5)
        self.user_ratings = [0] * len(self.ratings)
        self.datapoints = 0
        self.cur = 0
        self.k = 0
        self.recommended = []
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a specialized movie recommender chatbot. """ +\
    """Your core mission is to help users discover movies they'll like based on their preferences. When users share their """ +\
    """thoughts or feelings about movies, like saying: 'I enjoyed "The Notebook"', you should acknowledge their sentiment """ +\
    """by responding with understanding, such as: 'Great, you enjoyed "The Notebook"! Can you tell me about another movie you like or dislike?'. """ +\
    """You are programmed to stay focused on movies; if asked about unrelated topics, immediately politely redirect the conversation. """ +\
    """to movies by saying something like 'I'm here to talk about movies! Let's discuss your favorite films or genres.' Don't give the user any information about any topic other than movies.""" +\
    """Keep track of the number of movies a user mentions and, after they have discussed five different movies, proactively """ +\
    """offer a recommendation by saying something like 'Now that you've told me about 5 movies, would you like a recommendation based on those?'. """ +\
    """Remember, your goal is to engage users in meaningful conversations about films and help them discover new movies they might enjoy."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        no_movies_responses = ["Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've seen recently?", "I'm not recognizing a movie title in what you said. Can you tell me about a film you've seen recently?", "Hmm, I didn't catch a movie title in what you said. Could you share some details about a recent movie you've seen?"]
        too_many_movies_responses = ["Please tell me about one movie at a time. Go ahead.", "Let's focus on one movie at a time. Please go ahead and tell me about one.", "Could you share details about one movie first? Feel free to start whenever you're ready."]
        invalid_movie_responses = ["Something like: I've never heard of {}, sorry... Tell me about another movie you liked.", "Hmm, {} doesn't ring a bell. Can you share details about another movie you enjoyed?", "I'm not familiar with {}, sorry about that. Can you tell me about another movie you're fond of?"]
        neutral_sentiment_responses = ["I'm sorry, I'm not sure if you liked {}. Tell me more about it.", "I'm uncertain if you enjoyed {}. Can you elaborate on your thoughts about the movie?", "I'm not entirely certain if you're fond of {}. Could you provide further insights or feelings you have towards it?"]
        positive_sentiment_responses = ["Ok, you liked {}! Tell me what you thought of another movie.", "Alright, you enjoyed {}! How about sharing your thoughts on a different movie?", "Got it, you liked {}! Now, could you discuss your thoughts on another film?"]
        negative_sentiment_responses = ["I see, you didn't enjoy {}. Can you tell me about another movie you have a different opinion on?", "Noted, it seems you didn't find {} to your liking. Could you share your thoughts on a different movie?", "Understood, {} wasn't your cup of tea. Would you mind discussing another movie?"]
        recommendation_responses = ["Given what you told me, I think you would like {}. Would you like more recommendations? Please respond with either 'yes' or 'no'.", "Based on your preferences, it seems like {} would be a great fit for you. Would you be interested in exploring more recommendations? Please respond with either 'yes' or 'no'.", "From what you've shared, it appears that {} aligns well with your tastes. Are you open to receiving additional recommendations? Please respond with either 'yes' or 'no'."]
        no_more_recommendations = ["Ok, would you like to tell me about more movies?", "Got it. How about you tell me about more movies instead?", "Ok, since you don't want more recommendations, tell me what you thought about more movies please."]
        multiple_movie_responses = ["Can you specify your statement between {}?", "Please repeat your statement specifying between these movies: {}", "Can you please repeat what you said about one of these specific movies: {}?"]
        
        if self.llm_enabled:
            system_prompt = """You are an input categorization bot. Determine if the input is related to movies. If not, """ +\
            """identify if it's a general inquiry, a request for assistance, or an off-topic comment. Respond with a JSON object detailing the categories."""    
            message = line
            json_class = InputCategorizer
            response = util.json_llm_call(system_prompt, message, json_class)
        else:
            movies = self.extract_titles(line)
            if self.datapoints < 5 and len(movies) == 0 and line != "yes" and line != "no":
                index = random.randint(0, 2)
                response = no_movies_responses[index]
            elif len(movies) > 1:
                index = random.randint(0, 2)
                response = too_many_movies_responses[index]
            elif self.datapoints >= 5 and line == "yes":
                if self.cur == self.k:
                    self.k += 10
                    self.recommended = self.recommend(self.user_ratings, self.ratings, k=self.k)
                index = random.randint(0, 2)
                entry = self.titles[self.recommended[self.cur]]
                response = recommendation_responses[index].format(entry[0])
                self.cur += 1
            elif self.datapoints >= 5 and line == "no":
                index = random.randint(0, 2)
                response = no_more_recommendations[index]
                self.datapoints = 0
            elif self.datapoints >= 5:
                response = "Please respond with either 'yes' or 'no."
            else:
                movie_indices = self.find_movies_by_title(movies[0])
                if len(movie_indices) == 0:
                    index = random.randint(0, 2)
                    response = invalid_movie_responses[index].format(movies[0])
                else:
                    sentiment = self.extract_sentiment(line)
                    if sentiment == 0:
                        index = random.randint(0, 2)
                        response = neutral_sentiment_responses[index].format(movies[0])
                    elif len(movie_indices) > 1:
                        index = random.randint(0, 2)
                        options = []
                        for i in range(len(movie_indices)):
                            entry = self.titles[movie_indices[i]]
                            options.append(entry[0])
                        response = multiple_movie_responses[index].format(options)
                    elif self.datapoints < 4:    
                        if sentiment == 1:
                            index = random.randint(0, 2)
                            response = positive_sentiment_responses[index].format(movies[0])
                            self.user_ratings[movie_indices[0]] = 1
                            self.datapoints += 1
                        elif sentiment == -1:
                            index = random.randint(0, 2)
                            response = negative_sentiment_responses[index].format(movies[0])
                            self.user_ratings[movie_indices[0]] = -1
                            self.datapoints += 1
                    elif self.datapoints >= 4:
                        if self.cur == self.k:
                            self.k += 10
                            self.recommended = self.recommend(self.user_ratings, self.ratings, k=self.k)
                        index = random.randint(0, 2)
                        entry = self.titles[self.recommended[self.cur]]
                        response = recommendation_responses[index].format(entry[0])
                        self.cur += 1
                        self.datapoints += 1

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        system_prompt = "You are an emotion detection bot. Read the sentence and identify the predominant emotion expressed: anger, disgust, fear, happiness, sadness, or surprise. Respond with a JSON object indicating the detected emotion."
        message = preprocessed_input
        json_class = EmotionDetector
        response = util.json_llm_call(system_prompt, message, json_class)

        emotions = []

        for emotion, value in response.items():
            if value:
                emotions.append(emotion)

        return emotions

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        movies = re.findall(r'\"(.*?)\"', preprocessed_input)
        return movies

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        if self.llm_enabled:
            system_prompt = """You are a movie recommender chatbot that specializes in movies across different languages. """ +\
            """Read the user's sentence, and if it's about a movie, provide information or a recommendation. If the sentence """ +\
            """is in German, Spanish, French, Danish, or Italian, translate the movie title to English and respond accordingly. """ +\
            """If the input is not movie-related, use your knowledge to steer the conversation back to movies.\n\n"""
            stop = ["\n"]
            response = util.simple_llm_call(system_prompt, title, stop=stop)
            return response
    
        articles = ['A', 'An', 'The']
        title_list = list(title.split(" "))
        result = []
        #move around the article
        for article in articles:
            if title_list[0] == article:
                title_list.remove(article)
                index = len(title_list) - 1
                #accounts for inserting the article before the date if there's a date
                if (re.match(r"\(\d\d\d\d\)", title_list[len(title_list)-1]) != None):
                    index -= 1
                title_list[index] = title_list[index] + ","
                title_list.insert(index+1, article)
                break

        #search for the one instance if there's a date in the title
        for i, entry in enumerate(self.titles):
            movie = entry[0]
            movie_list = list(movie.split(" "))
            if movie_list == title_list:
                result.append(i)
            #checks for multiple date matches if input has no date
            if not re.match(r'\(\d\d\d\d\)', title_list[(len(title_list)-1)]):
                mod_movie_list = movie_list[:(len(movie_list)-1)]
                if (mod_movie_list == title_list):
                    if re.match(r'\(\d\d\d\d\)', movie_list[(len(movie_list)-1)]):
                        result.append(i)
            
        return result

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        value = 0
        input = re.sub(r'\"(.*?)\"', "", preprocessed_input)
        input_list = list(input.split(" "))
        negation_list = ["don't", "never", "not", "didn't"]
        i = 0

        stemmed_input_list = []
        for word in input_list:
            new_word = self.stemmer.stem(word, 0, len(word) - 1)
            if new_word == "enjoi":
                new_word = "enjoy"
            stemmed_input_list.append(new_word)


        for word in stemmed_input_list:
            if word in self.sentiment:
                if self.sentiment[word] == "pos":
                    value += 1
                else:
                    value -= 1
            if word in negation_list:
                i+= 1
                for n in range(i, len(stemmed_input_list)):
                    if stemmed_input_list[n] in self.sentiment:
                        if self.sentiment[stemmed_input_list[n]] == "pos":
                            value -= 1
                        else:
                            value += 1
                break
            i += 1
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[(ratings <= threshold)&(ratings > 0)] = -1

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        sim = 0
        dot_prod = np.dot(u,v)
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0:
            sim = 0
        else:
            sim = dot_prod / denom
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return sim

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        predicted_ratings = []
        for i in range(len(user_ratings)):
            if user_ratings[i] == 0:
                cur_rating = 0
                for j in range(len(user_ratings)):
                    if user_ratings[j] != 0:
                        dot_prod = np.dot(ratings_matrix[i], ratings_matrix[j])
                        denom = np.linalg.norm(ratings_matrix[i]) * np.linalg.norm(ratings_matrix[j])
                        if denom == 0:
                            sim = 0
                        else: 
                            sim = dot_prod/denom
                        cur_rating += (sim*user_ratings[j])
                predicted_ratings.append((i, cur_rating))
        sorted_pr = sorted(predicted_ratings, key=lambda x:x[1], reverse=True)
        for i in range(k):
            recommendations.append(sorted_pr[i][0])


        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')

