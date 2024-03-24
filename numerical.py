def find_best_split(self, data, attributes):
    entropy_of_full_set = self.entropy_of_split(data)
    best_attribute = ''
    max_information_gain = 0
    selected_attr_count = int(math.sqrt(len(attributes)))
    selected_attributes = random.sample(attributes, selected_attr_count)

    for attribute in selected_attributes:
        # unique_attribute_values = list(set(data[attribute]))
        if np.issubdtype(data[attribute].dtype, np.number):
            max_value = data[attribute].max()
            min_value = data[attribute].min()
            split_number = (max_value + min_value) / 2
            count_of_attribute_values = [0] * 2
            count_of_attribute_values[0] = len(data[data[attribute] <= split_number])
            count_of_attribute_values[1] = len(data[data[attribute] > split_number])
            # for i in range(len(2)):
            #     count_of_attribute_values[i] = len(data[data[attribute] == unique_attribute_values[i]])
            entropy_for_attribute = 0.0
            entropy_of_attribute_value = self.entropy_of_split(data[data[attribute] <= split_number])
            entropy_for_attribute = entropy_for_attribute + (count_of_attribute_values[0] * entropy_of_attribute_value)

            entropy_of_attribute_value = self.entropy_of_split(data[data[attribute] > split_number])
            entropy_for_attribute = entropy_for_attribute + (count_of_attribute_values[1] * entropy_of_attribute_value)

            # for i in range(len(2)):
            #     entropy_of_attribute_value = self.entropy_of_split(data[data[attribute] == unique_attribute_values[i]])
            #     entropy_for_attribute = entropy_for_attribute + (count_of_attribute_values[i] * entropy_of_attribute_value)
            entropy_for_attribute = entropy_for_attribute / sum(count_of_attribute_values)
            information_gain = entropy_of_full_set - entropy_for_attribute

            if (information_gain > max_information_gain):
                max_information_gain = information_gain
                best_attribute = attribute
    if max_information_gain == 0:
        return attributes[0], max_information_gain
    return best_attribute, max_information_gain