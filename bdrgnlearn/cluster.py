# Initialize a KMeans clustering class
class KMeans:
    # Define a method to initialize a class instance with the parameters
    def __init__(self, n_clusters=3, max_iter=300, n_init=10, eps=0.0001, lambd=1):

        # Pass parameters to class variables
        self.n_clusters_ = n_clusters
        self.max_iter_ = max_iter
        self.n_init_ = n_init
        self.eps_ = eps
        self.lambd_ = lambd

    # Initialize a method to assess the loss of clusters according to data
    def loss(self, cluster_centers):

        # Initialize an empty list to store the observations' cluster allignment
        cluster_allignment = []

        # Find alligned clusters
        for obs in self.X_train_:
            # Calculate distance of observation to all clusters
            obs_dist_clust = np.sum(np.power(cluster_centers - obs, 2), axis=1)

            # Find the nearest cluster index
            nearest_cluster_index = np.where(obs_dist_clust == np.min(obs_dist_clust))[0][0]

            # Append the nearest cluster index to the list of cluster allignment
            cluster_allignment.append(nearest_cluster_index)

            # Switch from cluster indexes to actual cluster coordinates
        nearest_cluster_coordinates = np.array([cluster_centers[x] for x in cluster_allignment])

        # Find the distance from each point to the nearest cluster
        distance_to_nearest_cluster = np.sum(np.power(self.X_train_ - nearest_cluster_coordinates, 2), axis=1)

        # Initialize a list to store the mean distance to the cluster center for each cluster
        mean_distance_from_center = []

        # Find the mean distance for each cluster
        for cluster in set(cluster_allignment):
            # Find the mean distance from observation to the cluster center
            mean_dist_for_cluster = np.mean(distance_to_nearest_cluster[np.array(cluster_allignment) == cluster])

            # Append to the list of mean distances
            mean_distance_from_center.append(mean_dist_for_cluster)

        # Calculate the final result
        total_distance = np.mean(mean_distance_from_center)

        # Return the total distance
        return total_distance

    # Define a method to fit the data
    def fit(self, X):

        # Save features values
        self.X_train_ = X

        # Initialize a variable to store the minimum loo
        minimum_loss = None

        # Iterate over number of initializations to find the best
        for i in tqdm(range(self.n_init_)):

            # Create an empty list to store randomly initialized cluster centers
            cluster_centers = []

            # Iterate over the number of clusters to fill the cluster_centers list with cluster coordinates
            for cluster in range(self.n_clusters_):

                # Randomly choose the starting point and append it to cluster_centers
                if len(cluster_centers) == 0:

                    # Randomly choose index from the features array
                    ind = np.random.randint(0, len(X))

                    # Randomly initialize the first cluster center
                    cluster_centers.append(X[ind, :])

                    # Find the point that is the most far from the current points
                else:

                    # Create variables to store the maximum distance, most far point and update
                    # if more far point is found
                    max_distance = None
                    most_far_point = None

                    # Iterate over all observation in search of the maximum distance from the current
                    # cluster centers
                    for obs_ind in range(len(self.X_train_)):

                        # Create a variable for observation according to the index
                        obs = self.X_train_[obs_ind, :]

                        # Find the euclidean distances to all cluster centers and sum them
                        distance = np.sum(np.sum(np.power(np.array(cluster_centers) - obs, 2), axis=1))

                        # If the point is more far than previous then change the most far point to current
                        if max_distance is None or max_distance < distance and (obs == cluster_centers).any() == False:
                            max_distance = distance
                            most_far_point = obs

                    # Append the most far point to cluster centers list
                    cluster_centers.append(most_far_point)

            # Conver the cluster centers list to array
            cluster_centers = np.array(cluster_centers).reshape(self.n_clusters_, self.X_train_.shape[1])

            # Initialize condition variables at starting points
            centers_change = True
            n_iter = 0

            # Continue movement towards cluster centers until maximum iterations limit is reached
            # or centers barely change (<= epsilon)
            while centers_change and n_iter <= self.max_iter_:

                # Increase a number of iterations to 1
                n_iter += 1

                # Initialize an empty list to store the observations' cluster allignment
                cluster_allignment = []

                # Find alligned clusters
                for obs in self.X_train_:
                    # Calculate distance of observation to all clusters
                    obs_dist_clust = np.sum(np.power(cluster_centers - obs, 2), axis=1)

                    # Find the nearest cluster index
                    nearest_cluster_index = np.where(obs_dist_clust == np.min(obs_dist_clust))[0][0]

                    # Append the nearest cluster index to the list of cluster allignment
                    cluster_allignment.append(nearest_cluster_index)

                    # Initialize a list to store the centroids' coordinates
                centroids = []

                # Find the means of the points in each cluster
                for cluster_index in range(self.n_clusters_):
                    # Initialize a boolean series of affiliation with certain cluster
                    cluster_affiliation = np.array(cluster_allignment) == cluster_index

                    # Refine the features array according to cluster affiliation
                    X_cluster = self.X_train_[cluster_affiliation]

                    # Find the mean point of refined array
                    mean_point = np.mean(X_cluster, axis=0)

                    # Append the mean point to the list of centroids
                    centroids.append(mean_point)

                # Reshape centroind according to the cluster centers format
                centroids = np.array(centroids)

                # Find the difference between centroids and current cluster centers
                needed_step = (centroids - cluster_centers)

                # Evaluate the stopping condition
                if np.sum(abs(needed_step)) < self.eps_:
                    centers_change = False

                # Make a step towards the centroids
                cluster_centers = cluster_centers + self.lambd_ * needed_step

        # Measure the current loss
        current_loss = self.loss(cluster_centers)

        # Compare current loss with minimum loss and replace if needed
        if minimum_loss is None or minimum_loss > current_loss:
            self.cluster_centers_ = cluster_centers
            minimum_loss = current_loss

    # Implement a method to predict cluster for a given point with a fitted model
    def predict(self, X):

        # Initialize an empty list to store the observations' cluster allignment
        cluster_allignment = []

        # Find alligned clusters
        for obs in X:
            # Calculate distance of observation to all clusters
            obs_dist_clust = np.sum(np.power(self.cluster_centers_ - obs, 2), axis=1)

            # Find the nearest cluster index
            nearest_cluster_index = np.where(obs_dist_clust == np.min(obs_dist_clust))[0][0]

            # Append the nearest cluster index to the list of cluster allignment
            cluster_allignment.append(nearest_cluster_index)

            # Return the cluster allignment for observations
        return cluster_allignment


