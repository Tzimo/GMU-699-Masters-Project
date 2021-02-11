import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

public class PDF {

	public static void pdf(File file, FileWriter writer, FileWriter writer2) throws IOException {

		boolean flag = false; //is there text in the document? default is no
		
		List<String> dates = new ArrayList<String>(); // ArrayList to hold the dates for each document
		PDDocument document = PDDocument.load(file); // Load the file
		PDFTextStripper stripper = new PDFTextStripper();
		stripper.setSortByPosition(true); // if columns, may be better to set to false
		try {
			for (int p = 1; p <= 2; ++p) // page interval to extract
			{
				stripper.setStartPage(p);
				stripper.setEndPage(p);
				String text = stripper.getText(document); // gets the text from the document
				// Regular Expressions used to look for each of the date formats
				String months = "((january|jan)|(february|feb)|(march|mar)|(april|apr)|may|(june|jun)|"				//months
						+ "(july|jul)|(august|aug)|(september|sep)|(october|oct)|(november|nov)|(december|dec))";
				String days_d = "(3[01]|[12][0-9]|0?[0-9])";	//EX: 30, 31, 05, 5
				String month_d = "(1[012]|0?[1-9])";	//EX: 12, 07, 4
				String year = "((19|20)\\d\\d|\\d\\d)"; // EX: 1998, 2003, 12, can also include any combination of 2 digits
				String regexp1 = "(" + months + " " + days_d + ", " + year + ")"; //EX: Jan 15, 2005 or January 3, 04
				String regexp2 = "(" + months + " " + year + ")";	//EX: Jan 2016
				String regexp3 = "(" + months + "  " + year + ")";	//EX: Jan  2016 (2 an extra space in between
				String regexp4 = "(" + month_d + "[/-]" + days_d + "[/-]" + year + ")";
				String regexp5 = "(" + "(19|20)\\\\d\\\\d" + month_d + days_d + ")";
				String regexp6 = "(" + year + "[/-]" + month_d + "[/-]" + days_d + ")";
				String regexp7 = "(" + days_d + " " + months + " " + year + ")";
				
				String regexp = "(?i)" + regexp1+"|"+regexp2+"|"+regexp3+"|"+regexp4+"|"+regexp5+"|"+regexp6+"|"+regexp7;
				Pattern pMod = Pattern.compile(regexp); // pattern to check against
				Matcher mMod = pMod.matcher(text); // check for matches in the text
				while (mMod.find()) { // loop through until there is no match
					dates.add(mMod.group(0)); // add the matched dates to the dates ArrayList
				}
			}
			
			String name = file.toString();
			while (name.toString().contains(",")) {
				StringBuilder new_name = new StringBuilder(name);
				new_name.setCharAt(name.indexOf(","), '_');
				name = new_name.toString();
			}
				
			writer.append(name + ",");
			
			if (dates.size()==0 & flag==true){ //if there was text in the document but it did not find a date
				SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
				writer.append(sdf.format(file.lastModified())+",Date Modified. Did not find a date in the text,");
			}
			
			if (dates.size()==0 & flag==false){ //There was not text in the document
				SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
				writer.append(sdf.format(file.lastModified())+",Date Modified. Did not find text,");
			}
			
			for (String date : dates) {
				if (date.contains(",")) {
					String new_date = date.replace(date, "\""+date+"\"");
					writer.append(new_date + ",");
				} else
					writer.append(date + ",");
			}
			writer.append("\n");
		} catch (IOException e) {
			SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
			writer2.append(file.toString() + ",File too large. Tried to be read but Not Read.   "+file.length()/1000000+",   "+sdf.format(file.lastModified()) + ",\n");
			//e.printStackTrace();
			System.out.println("Cought the error in PDF");
		} finally {
			document.close(); // Close the document
		}
	}
}
