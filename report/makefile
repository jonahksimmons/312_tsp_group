in_file=src/main.tex
out_dir=build
viewer=zathura

all: compile

compile: ${in_file}
	pdflatex -output-directory=${out_dir} ${in_file}
run:
	${viewer} ${out_dir}/*.pdf
clean:
	rm ${out_dir}/*

