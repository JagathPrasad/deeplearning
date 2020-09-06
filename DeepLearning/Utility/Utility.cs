using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace DeepLearning.Utility
{
    public class Utility
    {


        public dynamic LoadImages()
        {
            try
            {

            }
            catch (Exception ex)
            {

                throw ex;
            }
            return true;
        }

        public dynamic LoadData()
        {
            return true;
        }

        public dynamic LoadSmiles()
        {
            return true;
        }

        public Dictionary<string, int> ConvertTexttoInteger(char[] charList)
        {
            Dictionary<string, int> charToInteger = new Dictionary<string, int>();
            for (int i = 0; i < charList.Length; i++)
            {

                if (!charToInteger.ContainsKey(charList[i].ToString()))
                    charToInteger.Add(charList[i].ToString(), i);
            }
            return charToInteger;
        }

        public dynamic ConvertTexttoIntegerDynamic(char[] charList, Dictionary<string, int> dic)
        {
            int[] charToInteger = new int[charList.Length];
            for (int i = 0; i < charList.Length; i++)
            {
                try
                {
                    charToInteger[i] = dic.Where(x => x.Key.Equals(charList[i].ToString())).FirstOrDefault().Value;
                }
                catch (Exception ex)
                {
                }                
            }
            return charToInteger;
        }

        public Dictionary<int, string> ConvertIntegertoText(char[] charList)
        {
            Dictionary<int, string> intToChar = new Dictionary<int, string>();
            for (int i = 0; i < charList.Length; i++)
            {
                if (!intToChar.ContainsKey(charList[i]))
                    intToChar.Add(i, charList[i].ToString());
            }
            return intToChar;
        }
        public dynamic ConvertIntegertoTextDynamic(char[] charList)
        {
            Dictionary<int, string> intToChar = new Dictionary<int, string>();
            for (int i = 0; i < charList.Length; i++)
            {
                if (!intToChar.ContainsKey(charList[i]))
                    intToChar.Add(i, charList[i].ToString());
            }
            return intToChar;
        }
    }
}
