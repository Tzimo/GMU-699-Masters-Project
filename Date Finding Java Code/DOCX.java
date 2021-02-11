import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.apache.poi.xwpf.usermodel.XWPFDocument;

public class DOCX {

	public static XWPFWordExtractor extx = null;
	static boolean flag = false; //is there text in the document? default is no

	public static void docx(File file, FileWriter writer, FileWriter writer2) throws IOException {
		List<String> dates = new ArrayList<String>(); // ArrayList to hold the dates for each document
		try {
			FileInputStream fis = new FileInputStream(file.getAbsolutePath());
			XWPFDocument document = new XWPFDocument(fis);
			extx = new XWPFWordExtractor(document);
			String text = extx.getText();
			if (text != null) {
				String months = "((january|jan)|(february|feb)|(march|mar)|(april|apr)|may|(june|jun)|"
						+ "(july|jul)|(august|aug)|(september|sep)|(october|oct)|(november|nov)|(december|dec))";
				String days_d = "(3[01]|[12][0-9]|0?[0-9])";
				String month_d = "(1[012]|0?[1-9])";
				String year = "((19|20)\\d\\d|\\d\\d)";
				
				String regexp1 = "(" + months + " " + days_d + ", " + year + ")";
				String regexp2 = "(" + months + " " + year + ")";
				String regexp3 = "(" + months + "  " + year + ")";
				String regexp4 = "(" + month_d + "[/-]" + days_d + "[/-]" + year + ")";
				String regexp5 = "(" + "(19|20)\\\\d\\\\d" + month_d + days_d + ")";
				String regexp6 = "(" + year + "[/-]" + month_d + "[/-]" + days_d + ")";
				String regexp7 = "(" + days_d + " " + months + " " + year + ")";
				
				String regexp = "(?i)" + regexp1+"|"+regexp2+"|"+regexp3+"|"+regexp4+"|"+regexp5+"|"+regexp6+"|"+regexp7;
				Pattern pMod = Pattern.compile(regexp); // pattern to check against
				Matcher mMod = pMod.matcher(text); // check for matches in the text
				while (mMod.find()) { // loop through until there is no match
					//System.out.println(mMod.group(0));
					dates.add(mMod.group(0)); // add the matched dates to the dates ArrayList
				}
			}
			
			String name = file.toString();
			while (name.toString().contains(",")) {
				StringBuilder new_name = new StringBuilder(name);
				new_name.setCharAt(name.indexOf(","), '_');
				name = new_name.toString();
			}
			writer.append(name.toString() + ",");
			
			if (dates.size()==0 & flag==true){ //if there was text in the document but it did not find a date
				SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
				writer.append(sdf.format(file.lastModified())+",Date Modified. Did not find a date in the text,");
			}
			
			if (dates.size()==0 & flag==false){ //There was not text in the document
				SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
				writer.append(sdf.format(file.lastModified())+",Date Modified. Did not find text,");
			}
			
			for (int i=0; (i< dates.size() & i< 4);++i) {
				if (dates.get(i).contains(",")) {
					String date = dates.get(i).replace(dates.get(i), "\""+dates.get(i)+"\"");
					writer.append(date + ",");
				} else
					writer.append(dates.get(i) + ",");
			}
			writer.append("\n");
		} catch (Exception exep) {
			SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
			writer2.append(file.toString() + ",File too large. Tried to be read but Not Read.   "+file.length()/1000000+",   "+sdf.format(file.lastModified()) + ",\n");
			//exep.printStackTrace();
			System.out.println("Cought the error in DOCX");
		}
	}
}
