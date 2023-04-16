in_file=src/main.tex
ref_file=src/main
out_dir=build
viewer=zathura


compile: ${in_file}
	pdflatex -output-directory=${out_dir} ${in_file}
bib: ${in_file}
	pdflatex -output-directory=${out_dir} ${in_file}
	bibtex ${ref_file}
	pdflatex -output-directory=${out_dir} ${in_file}
	pdflatex -output-directory=${out_dir} ${in_file}
run:
	${viewer} ${out_dir}/*.pdf
clean:
	rm ${out_dir}/*

