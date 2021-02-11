
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class XLSX {

    public static void xlsx(File file, FileWriter writer, FileWriter writer2) throws IOException {
    	try{
    	List<String> dates = new ArrayList<String>(); // ArrayList to hold the dates for each document
            FileInputStream excelFile = new FileInputStream(file);
            Workbook workbook = new XSSFWorkbook(excelFile);
            Sheet datatypeSheet = workbook.getSheetAt(0);
            Iterator<Row> iterator = datatypeSheet.iterator();

            while (iterator.hasNext()) {

                Row currentRow = iterator.next();
                Iterator<Cell> cellIterator = currentRow.iterator();

                while (cellIterator.hasNext()) {

                    Cell currentCell = cellIterator.next();
                    if (currentCell.getCellType() == CellType.NUMERIC && currentCell.getDateCellValue()!= null) {
                    	//System.out.println(currentCell.getDateCellValue().toString()+file);
                       String text = currentCell.getDateCellValue().toString().substring(4,10)+", "+currentCell.getDateCellValue().toString().substring(24,28);
                       if (!text.contains("1900")&!text.contains("1899")&!text.contains("1901")&!text.contains("1902")) {
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
               					dates.add(mMod.group(0)); // add the matched dates to the dates ArrayList
               				}
               			}
                       }
                    }
                }
            }
            String name = file.toString();
            SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
			while (name.toString().contains(",")) {
				StringBuilder new_name = new StringBuilder(name);
				new_name.setCharAt(name.indexOf(","), '_');
				name = new_name.toString();
			}
			writer.append(name.toString() + ",");
			writer.append(sdf.format(file.lastModified())+",");
			writer.append("\n");
			workbook.close();
    	}catch (Exception exep) {
			SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy");
			writer2.append(file.toString() + ",File too large. Tried to be read but Not Read.   "+file.length()/1000000+",   "+sdf.format(file.lastModified()) + ",\n");
			//exep.printStackTrace();
			System.out.println("Cought the error in XLSX");
		}
    }
}