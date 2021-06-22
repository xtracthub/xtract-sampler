
with open('<name_of_old_file.csv>', 'r') as f:
    for line in f:
        print(line)
        with open('<name_of_new_files.csv>', 'a') as g:
            new_line = line.replace('/Users/ryan/Documents/CS/CDAC/official_xtract/sampler_dataset/', '<path_to_pub_8_on_matts_computer>')
            g.write(new_line)