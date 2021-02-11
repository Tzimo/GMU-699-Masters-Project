import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

import javax.swing.*;
import java.awt.event.*;
import java.awt.*;

class Main extends JFrame {

	public static FileWriter writer = null;
	public static FileWriter writer2 = null;

	static List<File> allfiles = new ArrayList<File>();

	public Main() {

		setLocation(600, 500);
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		JPanel jp = new JPanel(new GridLayout(5, 1));
		JLabel lbl = new JLabel(
				"<html>Enter File Direcory Path to Start Document Search  <br/>"
				+ "(Example: C:/Users/.....)  <br/> "
				+ "Do not close the window until notified:  </html>");
		final JTextField tf1 = new JTextField(10);
		final JTextField tf2 = new JTextField(10);
		final JTextField tf3 = new JTextField(10);
		JButton btn = new JButton("Click to Extract the Dates");
		btn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				
				try {
					File dir = new File(tf1.getText()); // directory where the files are
					String loc = "FoundandReadFiles.csv"; // location of Exported CSV for the found and read files
					String loc2 = "FilesNotRead.csv"; // location of Exported CSV for the file formats that do not match
					writer = new FileWriter(loc, true); //
					writer.append("DOCUMENT NAME,DATE LIST,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,\n");

					writer2 = new FileWriter(loc2, true);
					writer2.append("DOCUMENT NAME,NOTES,DATE LAST MODIFIED,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,---,\n");

					for (File file : getFiles(dir.listFiles())) { // loop through the files in the directory
						try {
							if (file.length() / 1000000 < 10) {

								if (file.toString().contains(".pdf") | file.toString().contains(".PDF")) {
									PDF.pdf(file, writer, writer2);
								} else if (file.toString().contains(".docx")) {
									DOCX.docx(file, writer, writer2);
								} else if (file.toString().contains(".doc")) {
									DOC.doc(file, writer, writer2);
								} else if (file.toString().contains(".pptx")) {
									PPTX.pptx(file, writer, writer2);
								} else if (file.toString().contains(".ppt")) {
									PPT.ppt(file, writer, writer2);
								} else if (file.toString().contains(".xlsx")) {
									XLSX.xlsx(file, writer, writer2);
								} else if (file.toString().contains(".xls")) {
									XLS.xls(file, writer, writer2);
								} else {
									SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
									writer2.append(file.toString() + ",File not read or checked,"
											+ sdf.format(file.lastModified()) + ",\n");
									//System.out.println("Wrong file format in Main");
								}
							} else {
								SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
								writer2.append(file.toString() + ",File too large." + file.length() / 1000000 + ","
										+ sdf.format(file.lastModified()) + ",\n");
								//System.out.println("Too large in Main");
							}

						} catch (Exception e) {
							SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
							writer2.append(file.toString() + ",Checked but wrong file format and caught by exception,"
									+ sdf.format(file.lastModified()) + ",\n");
							//System.out.println("caught the error in Main");
						}
						
					}
					tf2.setText("Finished.You may now close the window.");
					writer.close();
					writer2.close();

				} catch (Exception exep) {
					exep.printStackTrace();
					tf3.setText("Please Enter a Valid File Directory.");
				}
			}
		});

		jp.add(lbl);
		jp.add(tf1);
		jp.add(btn);
		jp.add(tf3);
		jp.add(tf2);
		getContentPane().add(jp);
		pack();

	}

	public static void main(String[] args) {
		new Main().setVisible(true);

	}

	public static List<File> getFiles(File[] files) {

		for (File file : files) { // Loop through the files in a directory to get a list of all the files, even
									// within other directories
			if (file.isDirectory()) {
				getFiles(file.listFiles()); // recursion to get to the lowest level of files
			} else {
				allfiles.add(file);
			}
		}
		return allfiles;
	}
}
