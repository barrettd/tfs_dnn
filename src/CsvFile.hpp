//
//  CsvFile.hpp
//
//  Created by Barrett Davis on 6/12/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//
#include "Matrix.hpp"

#ifndef CsvFile_h
#define CsvFile_h

namespace tfs {         // Tree Frog Software

    // ------------------------------------------------------------------------------------------------
    // Some data files contain no entry for a column.  e.g. "1,2,3,,5,6"
    // When the input data has no data, you can choose what data to use as a filler.
    // By default, we choose the "Not A Number" value (NAN), that can be tested for using isnan().
    // We assume that the file has a header row and all of the data values can be interpreted as double.
    // ------------------------------------------------------------------------------------------------
    Matrix *csvReadFile( const char *path, const double notPresent = NAN );

}   // namespace tfs

#endif /* CsvFile_h */
