# My version
def word_generation_model(data_x, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27, verbose = False):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data_x -- text corpus, divided in words
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    #parameters = initialize_parameters(n_a, n_x, n_y)
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    # Initialize loss (this is required because we want to smooth our loss)
    seq_length = dino_names
    loss = -np.log(1.0/vocab_size)*seq_length
    
    # Build list of all dinosaur names (training examples).
    examples = [i.strip() for i in data_x]
    
    n_e = len(examples)
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # for grading purposes
    last_dino_name = "abc"
    
    j_prev = 0
    
    # This is the number of iterations to change each character in a word, so the value should be at least
    # the length of the word (we change/predict one character per iteration)
    # We put the length of the vocabulary size because this is a large number, we will probably need less iterations
    T_x_max = n_x
    
    # Initialize the full 3D matricies for forward propagation
    # They store computed information ONLY - why save it? Maybe delete
    x = np.zeros((n_x, n_e, T_x_max))   # this contains the new created words
    a_prev = np.zeros((n_a, n_e, T_x_max))
    a_next = np.zeros((n_a, n_e, T_x_max))
    y_hat = np.zeros((n_y, n_e, T_x_max))
    
    # Initialize the full 3D matricies for backward propagation
    # They store computed information ONLY - why save it? Maybe delete
    dWax = np.zeros((n_a, n_x, T_x_max))
    dWaa = np.zeros((n_a, n_a, T_x_max))
    dWya = np.zeros((n_y, n_a, T_x_max))
    db = np.zeros((n_a, 1, T_x_max))
    dby = np.zeros((n_y, 1, T_x_max))
    da_next = np.zeros((n_a, n_e, T_x_max))
    dy = np.zeros((n_y, n_e, T_x_max))
    
    learning_rate = 0.01
    
    
    # Optimization loop
    for j in range(num_iterations):
        
        # ----------------------------
        #if (j > n_e-1) & (j % (n_e) == 0):
            # Restart count
        #    j_prev = j
            #print('j : ' + str(j) + ', idx : ' + str(j - j_prev) + ', j_prev : ' + str(j_prev))
        #idx = j - j_prev
        # OR
        idx = j % len(examples)
        #print('1. loop iteration per word : idx = ' + str(idx))
        # ----------------------------
        
        # ----------------------------
        # Set the input X (see instructions above)
        single_example = examples[idx]
        #print('single_example = ' + str(single_example))
        
        # Convert a string into a list of characters
        single_example_chars = [c for c in single_example]
        #print('2. word characters : single_example_chars : ' + str(single_example_chars))

        # Convert the list of characters to a list of integers
        single_example_ix = [char_to_ix[single_example_chars[i]] for i in range(len(single_example_chars))]
        #print('3. index of words : single_example_ix : ' + str(single_example_ix))
        # ----------------------------
       
        # ----------------------------
        # sorted_single_example_ix = np.unique(np.sort(single_example_ix))
        
        # Way 1 : Putting zeros : not correct because the 0 entries are used in the learning (DID NOT WORK)
        #X = np.zeros((T_x_max, 1))  #Y is the length of T_x
        #cout = 0
        #for i in range(T_x_max):
        #    if cout < len(sorted_single_example_ix):
        #        if i == sorted_single_example_ix[cout]:
        #            X[i] = int(sorted_single_example_ix[cout])
        #            cout = cout + 1
                    
        # Way 2: Putting the word entries without spaces - in the sequencial order of the word
        X = single_example_ix
        
        # Way 3: Putting None (DID NOT WORK)
        #X = []
        #cout = 0
        #for i in range(T_x_max):
        #    if cout < len(sorted_single_example_ix):
        #        if i == sorted_single_example_ix[cout]:
        #            X.append(int(sorted_single_example_ix[cout]))
        #            cout = cout + 1
        #        else:
        #            X.append(None)
        #    else:
        #        X.append(None)
        
        X = np.ravel(X)
        #print('length X : ' + str(len(X)))
        #print('4. set X : X : ' + str(X))
        # ----------------------------
        
        # ----------------------------
        ix_newline = char_to_ix['\n']
        
        # Way 1 : Putting zeros : not correct because the 0 entries are used in the learning (DID NOT WORK)
        #Y = np.zeros((T_x_max, 1))
        #for i in range(0,T_x_max-1):
        #    Y[i] = X[i+1]
        #Y[T_x_max-1] = ix_newline
        
        # Way 2: Putting the word entries without spaces - in the sequencial order of the word
        len_X = len(X)
        Y = []
        for i in range(0,len_X-1):
            Y.append(X[i+1])
        Y.append(ix_newline)
        
        # Way 3: Putting None (DID NOT WORK)
        #Y = []
        #for i in range(0,T_x_max-1):
        #    Y.append(X[i+1])
        #Y.append(ix_newline)
        
        Y = np.ravel(Y)
        #print('length Y : ' + str(len(Y)))
        #print('4. set Y : Y : ' + str(Y))
        # ----------------------------
        
        #if (idx == 0): #  "...saurus" example
        #if (idx == ):  #  "...tor" example
        #if (idx == ):  #  "...nia" example
        #    print('1. loop iteration per word : idx = ' + str(idx))
        #    print('2. word characters : single_example_chars : ' + str(single_example_chars))
        #    print('3. index of words : single_example_ix : ' + str(single_example_ix))
        #    print('4. set X : X : ' + str(X))
        #    print('4. set Y : Y : ' + str(Y))
        
        
        # ----------------------------
        # Forward propagate through time (≈1 line)
        loss = 0  # initialize your loss to 0
        indices = []
        created_char = []
        flag = 0
        t = 0
        samp_cout = 0
        
        loc_of_char = single_example_ix[0]   # Initialization only, should be never used
        
        # Initialize first iteration step of x_slice
        x_slice = x[:,idx:(idx+1),0:1] # 1 in the 2nd entry because there is one example word at a time
        x_slice = np.reshape(x_slice, (n_x,1))
        
        # For the first iteration you put zeros,
        # but for the iterations after you put the last a_prev_slice iteration from the previous forward prop
        a_prev_slice = a_prev[:,idx:(idx+1),0:1]
        a_prev_slice = np.reshape(a_prev_slice, (n_a,1))
        
        #print('5. START FORWARD PROP ')
        
        while (flag  == 0) & (t < len_X):  # for Way 2, similar to sample (BEST)
        #for t in range(len_X):   # for Way 2, original code (NOT EFFECTIVE : learns "\n" as part of the word)
        #while (flag  == 0) & (t < T_x_max):  # for Way 1 and 3 (DID NOT WORK)
        #for t in range(T_x_max):   # for Way 1 and 3, original code (DID NOT WORK)
            
            # ---------Initialize x for each character prediction in a word-----------------
            
            
            # Way 1 : Putting zeros : DID NOT WORK because the 0 entries are used in the learning
            #if int(X[t]) != 0:
            #    print('6. set x : X[t] = ' + str(int(X[t])))
            #    x_slice[int(X[t])] = 1  # Accurate location of character
            #else:
            #    print('6. set x : loc_of_char = ' + str(loc_of_char))
            #    x_slice[loc_of_char] = 1   # Predicted location of character from previous iteration
            
            #OR
            
            # Way 2: Putting the word entries without spaces - in the sequencial order of the word
            # Maximal similarity of single_example_ix and x : give all characters of single_example_ix to x
            word_len_cut = 0
            max_word_sample = len(X) - word_len_cut  # entire word or less
            #print('6. max_word_sample : ' + str(max_word_sample))
            
            if (samp_cout < max_word_sample):
                #print('6. set x : X[samp_cout] = ' + str(X[samp_cout]))
                x_slice[X[samp_cout]] = 1  # Accurate location of character
            else:
                # You use predicted locations when you do not have any more original word samples to give
                # This is the algorithm learning on it's own!
                #print('6. set x : loc_of_char = ' + str(loc_of_char))
                x_slice[loc_of_char] = 1   # Predicted location of character from previous iteration
            
            # OR
            
            # Way 3: Putting None : DID NOT WORK (same result as putting 0 - also can not know which Y[t] to output for loss)
            # if X[t] != None:
            #     print('6. set x : X[t] = ' + str(X[t]))
            #     x_slice[X[t]] = 1  # Accurate location of character
            # else:
            #     print('6. set x : loc_of_char = ' + str(loc_of_char))
            #     x_slice[loc_of_char] = 1   # Predicted location of character from previous iteration
            
            
            # --------------------------
            
            # ----------Forward propagation RNN----------------
            # Run one step forward of the RNN
            a_next_slice = np.tanh(np.dot(Waa, a_prev_slice) + np.dot(Wax, x_slice) + b)
            y_hat_slice = softmax(np.dot(Wya, a_next_slice) + by)
            # --------------------------
            
            # ---------Calculate the loss/error-----------------
            # Update the loss by substracting the cross-entropy term of this time-step from it.
            # loss is large when y_hat is NOT on the peak of the probablity distribution of Y
            # loss is small when y_hat is on the peak of the probablity distribution of Y (ie : log(1) = 0)
            loss = loss - np.log(np.ravel(y_hat_slice[Y[t]]))   #this means that Y[t] always has a relavant value
            
            #if (idx == 0): #  "...saurus" example
            #if (idx == ):  #  "...tor" example
            #if (idx == ):  #  "...nia" example
            #    print('7. loss calculated : loss = ' + str(loss))
            # --------------------------
            
            # ----------Predict the character----------------
            # In order to use random.choice you need y to be a list
            loc_of_char = np.random.choice(range(n_y), p = np.ravel(y_hat_slice))
            indices = indices + [loc_of_char]
            created_char = created_char + [ix_to_char[loc_of_char]]
            # --------------------------
            
            # -----------Update slices and save to 3D matrices---------------
            # Update x_slice and a_prev_slice
            x[0:n_x,idx:(idx+1),t:(t+1)] = np.reshape(x_slice, (n_x, 1, 1)) # Pass the x_slice to x for saving
            x_slice = np.zeros((n_x, 1))  # Reset x_slice to zero for the next character in word
            
            a_prev[0:n_a,idx:(idx+1),t:(t+1)] = np.reshape(a_prev_slice, (n_a, 1, 1)) # Pass the a_prev_slice to a_prev for saving
            a_prev_slice = a_next_slice  # Assign a_next_slice to a_prev_slice
            
            # Pass a_next_slice and y_hat_slice to their respective 3D matricies for saving
            a_next[0:n_a,idx:(idx+1),t:(t+1)] = np.reshape(a_next_slice, (n_a, 1, 1))
            y_hat[0:n_y,idx:(idx+1),t:(t+1)] = np.reshape(y_hat_slice, (n_y, 1, 1))
            # --------------------------
            
            # ------------Break/increment the loop--------------
            #print('8. decision for continuing the loop : indices[t] = ' + str(indices[t]))
            if indices[t] == char_to_ix["\n"]:
                # Break the loop
                flag = 1
            else:
                t = t + 1
                samp_cout = samp_cout + 1
            # --------------------------
        #print('9. FORWARD PROP DONE')
        
        #if (idx == 0): #  "...saurus" example
        #if (idx == ):  #  "...tor" example
        #if (idx == ):  #  "...nia" example
        #    print('indices : ' + str(indices)) 
        #    print('created_char : ' + str(created_char))
        #    print('----------------------------')
        # ----------------------------
        
        # ----------------------------
        # Backpropagate through time/iterations used in forward propagation (≈1 line)
        #print('10. BACKWARD PROP STARTING')
        
        # Initialize the slice
        dWya_slice = np.zeros_like(Wya)
        dby_slice = np.zeros_like(by)
        db_slice = np.zeros_like(b)
        dWax_slice = np.zeros_like(Wax)
        dWaa_slice = np.zeros_like(Waa)
        
        # Backpropagate through time
        counter = t   #this is the number of iterations needed in forward prop for a word
        #print('counter : ' + str(counter))
        for t in reversed(range(counter)):    #for t in reversed(range(T_x)):
            
            # ------------Update saved forward propagation slice values--------------
            a_next_slice = a_next[0:n_a,idx:(idx+1),t:(t+1)]
            a_next_slice = np.reshape(a_next_slice, (n_a,1))
            
            a_prev_slice = a_prev[0:n_a,idx:(idx+1),t:(t+1)]
            a_prev_slice = np.reshape(a_prev_slice, (n_a,1))
            
            da_next_slice = da_next[0:n_a,idx:(idx+1),t:(t+1)]
            da_next_slice = np.reshape(da_next_slice, (n_a,1))
            
            x_slice = x[0:n_x,idx:(idx+1),t:(t+1)]
            x_slice = np.reshape(x_slice, (n_x,1))
            # --------------------------
            
            # -----------Backpropagation---------------
            dy_slice = y_hat[0:n_y,idx:(idx+1),t:(t+1)]
            dy_slice = np.reshape(dy_slice, (n_y,1))
            dy_slice[Y[t]] = dy_slice[Y[t]] - 1  # This makes character value locations small
            
            dWya_slice += np.dot(dy_slice, a_next_slice.T)
            dby_slice += dy_slice
            da_slice = np.dot(Wya.T, dy_slice) + da_next_slice # backprop into h
            daraw = (1 - a_next_slice * a_next_slice) * da_slice # backprop through tanh nonlinearity
            db_slice += daraw
            dWax_slice += np.dot(daraw, x_slice.T)
            dWaa_slice += np.dot(daraw, a_prev_slice.T)
            da_next_slice = np.dot(Waa.T, daraw)
            
            # OR
            
            
            # --------------------------
            
            # -----------Save to 3D matrices---------------
            dWya[0:n_y,0:n_a,t:(t+1)] = np.reshape(dWya_slice, (n_y, n_a, 1))
            dby[0:n_y,0:1,t:(t+1)] = np.reshape(dby_slice, (n_y, 1, 1))
            db[0:n_a,0:1,t:(t+1)] = np.reshape(db_slice, (n_a, 1, 1))
            dWax[0:n_a,0:n_x,t:(t+1)] = np.reshape(dWax_slice, (n_a, n_x, 1))
            dWaa[0:n_a,0:n_a,t:(t+1)] = np.reshape(dWaa_slice, (n_a, n_a, 1))
            da_next[0:n_a,idx:(idx+1),t:(t+1)] = np.reshape(da_next_slice, (n_a, 1, 1))
            # --------------------------
        # ----------------------------
            
        # ----------------------------
        # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
        #gradients = clip(gradients, 5)
        maxValue = 3
        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient, -maxValue, maxValue, out = gradient)
        # ----------------------------
        
        # -------------Update the original random weight matrices with the last backproagation slice---------------
        # Update parameters (≈1 line)
        #parameters = update_parameters(parameters, gradients, learning_rate)
        # ***** NOTE: learning ACROSS word iterations are in these parameters! *****
        Wax = Wax - learning_rate*dWax_slice
        Waa = Waa - learning_rate*dWaa_slice
        Wya = Wya - learning_rate*dWya_slice
        b = b - learning_rate*db_slice
        by = by - learning_rate*dby_slice
        # ----------------------------
        #print('11. BACKWARD PROP DONE')
        
        
        curr_loss = loss
        
        # -----Give a_prev_slice to intialize the next word : this makes the learning ACROSS word iterations!------
        #gradients = gradients
        #a_prev = a_backprop[len(X)-1]
        # a_prev[:,idx,T_x] = a_next[:,idx,T_x-1]
        
        # Carry a_prev to next word (matrix entry for idx less than example size length)
        # a_prev[0:n_a,(idx+1):(idx+2),0:1] = np.reshape(a_next_slice, (n_a, 1, 1))  (DID NOT WORK)
        # OR
        # Do not carry over a_prev to the next word  (BEST: seems to learn chunks of words better like "saurus")
        # ----------------------------
        
        # debug statements to aid in correctly forming X, Y
        print_it = 1
        if print_it == 1:
            if verbose and j in [0, len(examples) -1, len(examples)]:
                print("j = " , j, "idx = ", idx,) 
            if verbose and j in [0]:
                print("single_example =", single_example)
                print("single_example_chars", single_example_chars)
                print("single_example_ix", single_example_ix)
                print(" X = ", X, "\n", "Y =       ", Y, "\n")

            # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
            loss = smooth(loss, curr_loss)

            # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
            if j % 2000 == 0:

                print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

                # The number of dinosaur names to print
                seed = 0
                for name in range(dino_names):

                    # Sample indices and print them
                    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
                    sampled_indices = sample(parameters, char_to_ix, seed)
                    last_dino_name = get_sample(sampled_indices, ix_to_char)
                    print(last_dino_name.replace('\n', ''))

                    seed += 1  # To get the same result (for grading purposes), increment the seed by one. 

                print('\n')
       
        
    return parameters, last_dino_name