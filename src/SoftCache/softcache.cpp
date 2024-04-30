#include <softcache.hpp>

using namespace std;

#if DEBUG
    #define dout cout
#else
    #define dout 0 && cout
#endif


Cache::Cache(int argc, char** argv)
{
    InputParser input(argc, argv);
    Organisation org = DIRECT_MAPPING;
    ReplacementPolicy rp = LRU;
    int cacheSize = 0;
    int linesPerSet = 1;
    bool write_back = false;

    const std::string &orgString = input.getCmdOption("-o");
    const std::string &rpString = input.getCmdOption("-r");
    const std::string &cacheSizeString = input.getCmdOption("-c");
    const std::string &linesPerSetString = input.getCmdOption("-l");
    const std::string &writeBackString = input.getCmdOption("-w");

    if (!orgString.empty() && !rpString.empty() && !cacheSizeString.empty()){
        cout << cacheSizeString << endl;
        cacheSize = atoi(cacheSizeString.c_str());

        if (orgString == "d" || orgString == "direct_mapping") {
            org = DIRECT_MAPPING;
            linesPerSet = 1;
        } else if (orgString == "s" || orgString == "set_associative") {
            org = SET_ASSOCIATIVE;
            if (!linesPerSetString.empty()){
                linesPerSet = atoi(linesPerSetString.c_str());
            } else {
                cout << "Invalid lines per set" << endl;
                exit(1);
            }
        } else if (orgString == "f" || orgString == "fully_associative") {
            org = FULLY_ASSOCIATIVE;
        } else {
            cout << "Invalid organisation" << endl;
            exit(1);
        }

        if (rpString == "lru") {
            rp = LRU;
        } else if (rpString == "fifo") {
            rp = FIFO;
        } else if (rpString == "random") {
            rp = RANDOM;
        } else if (rpString == "smallest") {
            rp = SMALLEST;
        } else if (orgString != "d" && orgString != "direct_mapping") {
            cout << "Invalid replacement policy, using LRU" << endl;
            exit(1);
        }

        if (writeBackString == "10") { // Write-through
            write_back = false;
        } else if (writeBackString == "01") { // Write-back
            write_back = true;
        }
    }

    initialise(org, rp, cacheSize, linesPerSet, write_back);
}

/*! 
    * \brief Constructor
    * \param organisation DIRECT MAPPING, SET ASSOCIATIVE, FULLY ASSOCIATIVE
    * \param replacementPolicy LRU, FIFO, RANDOM
    * \param cacheSize The size of the cache
    * \param linesPerSet The number of lines per set. Only used for set associative caches.
    */
Cache::Cache(Organisation organisation, ReplacementPolicy replacementPolicy, int cacheSize, int linesPerSet, bool write_back) 
{
    initialise(organisation, replacementPolicy, cacheSize, linesPerSet, write_back);
}

Cache::~Cache()
{
    cout << "Cleaning up..." << endl;
    // Free all openCL objects
    cl_int err = 0;
    for (int i = 0; i < this->nrOfLines; ++i)
    {
        if (this->lines[i].deviceAddress != NULL)
        {
            buffers--;
            err |= clReleaseMemObject(this->lines[i].deviceAddress);
        }
        
    }

    if (err != CL_SUCCESS) 
    {
        printf("Error: Failed to release memory objects! %d", err);
    }

    // Free allocated memory for cache lines
    delete[] this->lines;
}

#if CACHE_ENABLED
/*! 
    * \brief Create a buffer in the cache
    * \param context The OpenCL context
    * \param flags The OpenCL flags
    * \param size The size of the buffer in bytes
    * \param host_ptr The host pointer
    * \param errcode_ret The error code
    * \return The device pointer
    */
cl_mem Cache::createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret)
{
    cl_mem deviceAddress;
    //START_TIMER
    if (flags & CL_MEM_COPY_HOST_PTR) 
    {
        this->duration.bytesTotal += size;
        this->duration.bytesh2d_total += size;

        CacheLine *cacheLine = getCacheLine(host_ptr);

        if (cacheLine == nullptr || cacheLine->flag == CPU)
        {
            this->duration.cacheMiss += 1;
            dout << "createBuffer: Cache miss" << endl;
            deviceAddress = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
            addToCache(host_ptr, size, deviceAddress, BOTH);
        } 
        else 
        {
            this->duration.cacheHit += 1;
            this->duration.bytesSaved += size;
            this->duration.bytesh2d_saved += size;
            int idx = (cacheLine - this->lines);
            this->lockedLines.push_back(idx);
            dout << "createBuffer: Cache hit on Line " << idx << endl;
            deviceAddress = cacheLine->deviceAddress;
        }
    } 
    else 
    {
        deviceAddress = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
        buffers++;
    }

    if (*errcode_ret != CL_SUCCESS || deviceAddress == nullptr)
    {
        printf("Error: Failed to create buffer! %p -> %s\n", deviceAddress, getErrorString(*errcode_ret).c_str());
    }
    //STOP_TIMER(this->duration.hostToDevice);
    return deviceAddress;   
}

/*! 
    * \brief Enqueue a write buffer command
    * \param command_queue The OpenCL command queue
    * \param buffer The buffer
    * \param blocking_write Blocking write
    * \param offset The offset
    * \param cb The size of the buffer in bytes
    * \param ptr The host pointer
    * \param event_wait_list The event wait list
    * \param event The event
    * \return The error code
    */
cl_int Cache::enqueueWriteBuffer(
    cl_command_queue command_queue, 
    cl_mem *buffer, cl_bool blocking_write, 
    size_t offset, 
    size_t cb, 
    const void *ptr, 
    cl_uint num_events_in_wait_list, 
    const cl_event *event_wait_list, 
    cl_event *event) 
{
    this->duration.bytesTotal += cb;
    this->duration.bytesh2d_total += cb;

    this->cache_command_queue = command_queue;
    //START_TIMER
    CacheLine *cacheLine = getCacheLine(ptr);

    if (cacheLine == nullptr)
    {
        this->duration.cacheMiss += 1;
        dout << "enqueueWriteBuffer: Cache miss" << endl;
        addToCache(ptr, cb, *buffer, BOTH);
    } 
    else if (cacheLine->flag == CPU)
    {
        this->duration.cacheMiss += 1;
        addToCache(ptr, cb, *buffer, BOTH, (cacheLine - this->lines));
    }
    else 
    {
        this->duration.cacheHit += 1;
        this->duration.bytesSaved += cb;
        this->duration.bytesh2d_saved += cb;

        // When a new buffer is created but there is already a buffer 
        // in the cache we can free the new one.
        if (cacheLine->deviceAddress != *buffer && *buffer != NULL)
        {
            buffers--;
            clReleaseMemObject(*buffer);
            *buffer = cacheLine->deviceAddress;
        }

        int idx = (cacheLine - this->lines);
        this->lockedLines.push_back(idx);
        dout << "enqueueWriteBuffer: Cache hit on Line " << idx << endl;
        return CL_SUCCESS;  // No need to write the buffer so return CL_SUCCESS
    }
    // On cache miss we have to write the buffer to the device
    cl_event myevent;
    cl_int err;
    err = clEnqueueWriteBuffer(
        command_queue, 
        *buffer, 
        blocking_write, 
        offset, 
        cb, 
        ptr, 
        num_events_in_wait_list, 
        event_wait_list, 
        &myevent
    );
    this->duration.hostToDevice += probe_event_time(myevent, command_queue);
    //STOP_TIMER(this->duration.hostToDevice);
    return err;
}

cl_int Cache::enqueueReadBuffer(
        cl_command_queue command_queue,
        cl_mem buffer, cl_bool blocking_read,
        size_t offset,
        size_t cb,
        void *ptr,
        cl_uint num_events_in_wait_list,
        const cl_event *event_wait_list,
        cl_event *event
    )
{
    this->lockedLines.clear();
    this->duration.bytesTotal += cb;
    this->duration.bytesSaved += cb; // Will subtract if transferred from device 
    this->duration.bytesd2h_total += cb;
    this->duration.bytesd2h_saved += cb;

    //START_TIMER
    this->cache_command_queue = command_queue;
    cl_int err = CL_SUCCESS;
    if (!write_back) 
    {
        cl_event myevent;
        err = clEnqueueReadBuffer(
            command_queue, 
            buffer, 
            blocking_read, 
            offset, 
            cb, 
            ptr, 
            num_events_in_wait_list, 
            event_wait_list, 
            &myevent
        );
        this->duration.deviceToHost += probe_event_time(myevent, command_queue);
        this->duration.bytesSaved -= cb;
        this->duration.bytesd2h_saved -= cb;

    }    

    CacheLine *cacheLine = getCacheLine(ptr);
    if (cacheLine == nullptr) 
    {
        dout << "enqueueReadBuffer: Cache miss" << endl;
        addToCache(ptr, cb, buffer, (!write_back && write_through) ? BOTH : GPU);
        //STOP_TIMER(this->duration.deviceToHost);
    } 
    else if (buffer != cacheLine->deviceAddress) 
    {
        // Cache hit, but we do have to remove the newly created buffer
        // otherwise it'll never be freed
        buffers--;
        clReleaseMemObject(buffer);
        //STOP_TIMER(this->duration.deviceToHost);
    }
    
    // Clear locked lines again...
    this->lockedLines.clear();

    return err;
}

#else

cl_mem Cache::createBuffer(
    cl_context context, 
    cl_mem_flags flags, 
    size_t size, 
    void *host_ptr, 
    cl_int *errcode_ret) 
{
    //START_TIMER
    if (flags & CL_MEM_COPY_HOST_PTR) 
    {
        this->duration.cacheMiss += 1;
        this->duration.bytesTotal += size;
    }
    cl_mem deviceAddress = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
    //STOP_TIMER(this->duration.hostToDevice)
    return deviceAddress;
}

cl_int Cache::enqueueWriteBuffer(
    cl_command_queue command_queue, 
    cl_mem *buffer, cl_bool blocking_write, 
    size_t offset, 
    size_t cb, 
    const void *ptr, 
    cl_uint num_events_in_wait_list, 
    const cl_event *event_wait_list, 
    cl_event *event) 
{
    this->duration.cacheMiss += 1;
    this->duration.bytesTotal += cb;
    //START_TIMER
    cl_event myevent;
    cl_int err = clEnqueueWriteBuffer(
        command_queue, 
        *buffer, 
        blocking_write, 
        offset, 
        cb, 
        ptr, 
        num_events_in_wait_list, 
        event_wait_list, 
        &myevent
    );
    this->duration.hostToDevice += probe_event_time(myevent, command_queue);
    //STOP_TIMER(this->duration.hostToDevice)
    return err;
}

cl_int Cache::enqueueReadBuffer(
        cl_command_queue command_queue,
        cl_mem buffer, cl_bool blocking_read,
        size_t offset,
        size_t cb,
        void *ptr,
        cl_uint num_events_in_wait_list,
        const cl_event *event_wait_list,
        cl_event *event,
        bool write_through
    )
{
    this->duration.bytesTotal += cb;
    //START_TIMER
    cl_event myevent;
    cl_int err = clEnqueueReadBuffer(
        command_queue, 
        buffer, 
        blocking_read, 
        offset, 
        cb, 
        ptr, 
        num_events_in_wait_list, 
        event_wait_list, 
        &myevent
    );
    this->duration.deviceToHost += probe_event_time(myevent, command_queue);
    //STOP_TIMER(this->duration.deviceToHost)
    return err;
}
#endif

cl_int Cache::setKernelArg(cl_kernel kernel, cl_uint index, size_t size, const void * value)
{
    kernelArguments[kernel].insert(value);
    return clSetKernelArg(kernel, index, size, value); 
}

cl_int Cache::enqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    this->lockedLines.clear();

    cl_event myevent;
    cl_int err = clEnqueueNDRangeKernel(
        command_queue, 
        kernel, 
        work_dim, 
        global_work_offset, 
        global_work_size, 
        local_work_size, 
        num_events_in_wait_list, 
        event_wait_list, 
        &myevent
    );   
    this->duration.kernel += probe_event_time(myevent, command_queue);

    auto& argumentsVector = kernelArguments[kernel];
    if (!argumentsVector.empty()) {
        // std::cout << "nr of args: " << argumentsVector.size() << endl;
        for (auto& argument : argumentsVector) 
        {
            setDirtyFlag(argument, GPU);
        }
    }

    return err;
}

cl_int Cache::writeBack()
{
    cl_int err = CL_SUCCESS;
#if CACHE_ENABLED
    if (this->write_back == false) return err;
    
    for (int i = 0; i < this->nrOfLines; ++i) 
    {
        if (this->lines[i].flag == GPU) 
        {
            cl_event myevent;
            err |= clEnqueueReadBuffer(
                this->cache_command_queue, 
                this->lines[i].deviceAddress, 
                CL_TRUE, 
                0, 
                this->lines[i].size, 
                this->lines[i].tag, 
                0, 
                NULL, 
                &myevent
            );
            this->duration.deviceToHost += probe_event_time(myevent, this->cache_command_queue);
            this->lines[i].flag = BOTH;
            this->duration.bytesSaved -= this->lines[i].size;
            this->duration.bytesd2h_saved -= this->lines[i].size;
        }
    }
#endif
    return err;
}

cl_int Cache::writeBack(void *host_ptr) 
{
    cl_int err = CL_SUCCESS;
#if CACHE_ENABLED
    if (this->write_back == false) return err;

    
    CacheLine *cacheLine = getCacheLine(host_ptr);
    if (cacheLine != nullptr && cacheLine->flag == GPU) 
    {
        cl_event myevent;
        err |= clEnqueueReadBuffer(
            this->cache_command_queue, 
            cacheLine->deviceAddress, 
            CL_TRUE, 
            0, 
            cacheLine->size, 
            cacheLine->tag, 
            0, 
            NULL, 
            &myevent
        );
        this->duration.deviceToHost += probe_event_time(myevent, this->cache_command_queue);
        cacheLine->flag = BOTH;
        this->duration.bytesSaved -= cacheLine->size;
        this->duration.bytesd2h_saved -= cacheLine->size;
    }
#endif
    return err;
}

void Cache::setDirtyFlag(const void *ptr, Flag flag)
{
    CacheLine *cacheLine = getCacheLine(ptr);
    if (cacheLine != nullptr)
    {
        cacheLine->flag = flag;
    }
}

void Cache::printCache()
{
    const string flags[] = {"CPU", "GPU", "BOTH"};

    printf("=============================================================================================\n");
    for (int i = 0; i < this->nrOfLines; ++i) 
    {
        if (this->organisation == SET_ASSOCIATIVE && i % this->nrOfLinesPerSet == 0) 
            printf("---------------------------------------------------------------------------------------------\n");

        printf("Line %-6d", i);
        printf("Flag: %-6s", flags[this->lines[i].flag].c_str());
        printf("Age: %-6d", this->lines[i].age);
        printf("Tag: %-18p", this->lines[i].tag);
        printf("Size: %-10lu", this->lines[i].size);
        printf("Device addr: %-18p", this->lines[i].deviceAddress);
        printf("\n");
    }
    const string organisationString[3] = {"DIRECT_MAPPING", "SET_ASSOCIATIVE", "FULLY_ASSOCIATIVE"};
    const string replacementPolicyString[4] = {"LRU", "FIFO", "RANDOM", "SMALLEST"};
    printf("%-30s %s\n", "Cache organisation:", organisationString[this->organisation].c_str());
    printf("%-30s %s\n", "Cache replacement policy:", replacementPolicyString[this->replacementPolicy].c_str());
    printf("%-30s %d\n", "Cache number of sets:", this->nrOfSets);
    printf("%-30s %d\n", "Cache number of lines:", this->nrOfLines);
    printf("=============================================================================================\n\n");
}

void Cache::printTimeProfile()
{
    printf("=========================================\n");
    printf("%-20s Time (ms)\n", "Action");
    printf("-----------------------------------------\n");
    printf("%-20s %llu\n", "Host to device", this->duration.hostToDevice / 1000);
    printf("%-20s %llu\n", "Device to host", this->duration.deviceToHost / 1000);
    printf("%-20s %llu\n", "Total on transfers", (this->duration.hostToDevice + this->duration.deviceToHost) / 1000);
    printf("%-20s %llu\n", "Kernel execution", this->duration.kernel / 1000);
    printf("%-20s %llu\n", "Total time ", (this->duration.hostToDevice + this->duration.deviceToHost + this->duration.kernel) / 1000);
    printf("-----------------------------------------\n");
    printf("%-20s %u\n", "Cache hits", this->duration.cacheHit);
    printf("%-20s %u\n", "Cache misses", this->duration.cacheMiss);
    printf("%-20s %.2f%%\n", "Hit ratio", (float) this->duration.cacheHit / (float)(this->duration.cacheHit + this->duration.cacheMiss) * 100);    
    printf("%-20s %zu\n", "Bytes saved", this->duration.bytesSaved);
    printf("%-20s %zu\n", "Bytes total", this->duration.bytesTotal);
    printf("%-20s %.2f%%\n", "byte ratio", (float) this->duration.bytesSaved / (float)(this->duration.bytesTotal) * 100);
    printf("%-20s %zu\n", "Bytes h2d saved", this->duration.bytesh2d_saved);
    printf("%-20s %zu\n", "Bytes h2d total", this->duration.bytesh2d_total);
    printf("%-20s %.2f%%\n", "byte h2d ratio", (float) this->duration.bytesh2d_saved / (float)(this->duration.bytesh2d_total) * 100);
    printf("%-20s %zu\n", "Bytes d2h saved", this->duration.bytesd2h_saved);
    printf("%-20s %zu\n", "Bytes d2h total", this->duration.bytesd2h_total);
    printf("%-20s %.2f%%\n", "byte d2h ratio", (float) this->duration.bytesd2h_saved / (float)(this->duration.bytesd2h_total) * 100);
    printf("=========================================\n");
}

void Cache::writeTimeProfileToFile(vector<string> other_info) 
{
    ofstream myfile ("log.txt", fstream::app);
    myfile.imbue(std::locale(std::cout.getloc(), new DecimalSeparator<char>(',')));
    if (myfile.is_open())
    {
        const string organisationString[3] = {"DIRECT_MAPPING", "SET_ASSOCIATIVE", "FULLY_ASSOCIATIVE"};
        const string replacementPolicyString[4] = {"LRU", "FIFO", "RANDOM", "SMALLEST"};

        // Timestamp YY-MM-DD hh:mm:ss
        myfile << currentDateTime() << " ";
        
        // Cache info
        myfile << organisationString[this->organisation] << " ";
        myfile << replacementPolicyString[this->replacementPolicy] << " ";
        myfile << this->nrOfSets << " ";
        myfile << this->nrOfLines << " ";       

        // h2d, kernel, d2h, total
        myfile << (this->duration.hostToDevice / 1000) << " " << (this->duration.deviceToHost / 1000) << " " << (this->duration.kernel / 1000) << " ";
        myfile << (this->duration.hostToDevice + this->duration.deviceToHost + this->duration.kernel) / 1000 << " "; // Total time

        // Cache hit, cache miss, cache hit ratio
        myfile << this->duration.cacheHit << " " << this->duration.cacheMiss << " ";
        myfile << ((float) this->duration.cacheHit / (float)(this->duration.cacheHit + this->duration.cacheMiss) * 100) << " "; // Cache hit ratio

        // Bytes saved, bytes total, byte hit ratio
        myfile << this->duration.bytesSaved << " " << this->duration.bytesTotal << " ";
        myfile << ((float) this->duration.bytesSaved / (float)(this->duration.bytesTotal) * 100) << " ";                        // Byte hit ratio

        // Bytes h2d saved, bytes h2d total, byte h2d ratio
        myfile << this->duration.bytesh2d_saved << " " << this->duration.bytesh2d_total << " ";
        myfile << ((float) this->duration.bytesh2d_saved / (float)(this->duration.bytesh2d_total) * 100) << " ";                        // Byte h2d ratio

        // Bytes d2h saved, bytes d2h total, byte d2h ratio
        myfile << this->duration.bytesd2h_saved << " " << this->duration.bytesd2h_total << " ";
        myfile << ((float) this->duration.bytesd2h_saved / (float)(this->duration.bytesd2h_total) * 100) << " ";                        // Byte d2h ratio

        // Whatever the client wants to add
        for (int i = 0; i < other_info.size(); ++i)
        {
            myfile << other_info[i] << " ";
        }

        myfile << endl;
        myfile.close();
    }
}

void Cache::resetTimers()
{
    this->duration.hostToDevice = 0;
    this->duration.deviceToHost = 0;
    this->duration.kernel = 0;
    this->duration.cacheHit = 0;
    this->duration.cacheMiss = 0;
    this->duration.bytesSaved = 0;
    this->duration.bytesTotal = 0;
    this->duration.bytesd2h_saved = 0;
    this->duration.bytesd2h_total = 0;
    this->duration.bytesh2d_saved = 0;
    this->duration.bytesh2d_total = 0;
}

void Cache::resetCache()
{
    cout << "Clearing cache..." << endl;
    cl_int err = 0;
    for (int i = 0; i < this->nrOfLines; ++i)
    {
        if (this->lines[i].deviceAddress != NULL)
        {
            buffers--;
            err |= clReleaseMemObject(this->lines[i].deviceAddress);
        }
        
    }

    if (err != CL_SUCCESS) 
    {
        printf("Error: Failed to release memory objects! %d", err);
    }

    memset(this->lines, 0, sizeof(CacheLine) * this->nrOfLines);
}


/* ===================== PRIVATE METHODS ===================== */


void Cache::initialise(Organisation organisation, ReplacementPolicy replacementPolicy, int cacheSize, int nrOfSet, bool write_back)
{
    this->organisation = organisation;
    this->replacementPolicy = replacementPolicy;

    if (organisation == DIRECT_MAPPING) 
    {
        // We use a hash function to compute the index of the set
        // therefore nrOfSets should be a prime number, otherwise we get a lot of collisions
        this->nrOfSets = getTableSize(cacheSize);
        this->nrOfLines = this->nrOfSets;
        this->nrOfLinesPerSet = 1;
        
    }
    else if (organisation == FULLY_ASSOCIATIVE) 
    {
        this->nrOfSets = 1;
        this->nrOfLines = cacheSize;
        this->nrOfLinesPerSet = cacheSize;
    } 
    else 
    {
        // We use a hash function to compute the index of the set
        // therefore nrOfSets should be a prime number, otherwise we get a lot of collisions
        this->nrOfSets = getTableSize(nrOfSet);
        this->nrOfLinesPerSet = cacheSize / this->nrOfSets;
        this->nrOfLines = this->nrOfSets * this->nrOfLinesPerSet;        
    }

    // debug info
    dout << "nrOfLines: " << nrOfLines 
         << "\t nrOfsets: " << nrOfSets 
         << "\t cacheSize: " << cacheSize 
         << endl;
    // assert(nrOfLines >= cacheSize);

    // Allocate memory for the cache lines and set to 0
    this->lines = new CacheLine[this->nrOfLines]();

    this->write_back = write_back;

    // Seed the random number generator
    srand ( time(NULL) );

    resetTimers();
    for (int i = 0; i < this->nrOfSets; ++i)
    {
        this->FIFO_index.push_back(0);
    }


    const string organisationString[3] = {"DIRECT_MAPPING", "SET_ASSOCIATIVE", "FULLY_ASSOCIATIVE"};
    const string replacementPolicyString[4] = {"LRU", "FIFO", "RANDOM", "SMALLEST"};
    printf("%-30s %s\n", "Cache organisation:", organisationString[this->organisation].c_str());
    printf("%-30s %s\n", "Cache replacement policy:", replacementPolicyString[this->replacementPolicy].c_str());
    printf("%-30s %d\n", "Cache number of sets:", this->nrOfSets);
    printf("%-30s %d\n", "Cache number of lines:", this->nrOfLines);
    printf("%-30s %s\n", "Write back:", this->write_back ? "true" : "false");
}

/*!
    * \brief Checks whether input integer is an odd prime number
    * \param n The input number
    * \return returns true if the input is an odd prime a number
    */
bool Cache::isPrime(int n)
{
    if (n <= 1) return false;
    if (n == 2) return false; // Technically a prime, but we dont want 2^n numbers
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

/*!
    * \brief The table size should be a prime number for direct mapping and set assiocative mapping.
    * This function will find the prime number that is greater or equal than the input number.
    * An additional constraint is that r % table_size is not equal to 1, where r is the number of possible addresses.
    * \param n The input number
    * \return The next prime number
    */
int Cache::getTableSize(int n)
{
    int prime = n;
    /* Additonal constraint: https://www.geeksforgeeks.org/what-are-hash-functions-and-how-to-choose-a-good-hash-function/
    It has been found that the best results with the division method are achieved when the table size is prime. 
    However, even if table_size is prime, an additional restriction is called for. 
    If r is the number of possible character codes on an computer, and if table_size is a prime such that r % table_size equal 1, 
    then hash function h(key) = key % table_size is simply the sum of the binary representation of the 
    characters in the key mod table_size. Therefore, many keys can have the same hash.
    */
    // while(!isPrime(prime) || ((uintptr_t) ~0 % prime) == 1) prime--;
    if (n <= 2) return 3;
    if (isPrime(n)) return n;
    int lower = n - 1;
    int upper = n + 1;
    while (true) {
        if (isPrime(lower) && ((uintptr_t) ~0 % prime) != 1) return lower;
        if (isPrime(upper) && ((uintptr_t) ~0 % prime) != 1) return upper;
        lower--;
        upper++;
    }
    return prime;
}

/*!
    * \brief Get the set index
    * \param tag The tag
    * \return The set index
    */
int Cache::getSetIndex(const void *tag)
{
    int setIdx = ((uintptr_t) tag) % nrOfSets;
    return setIdx;
}

/*!
    * \brief Checks whether the tag is in the respective set
    * \param setIndex The set index
    * \param tag The tag    
    * \return Returns a pointer to the respective cache line or NULL if it does not exist
    */
CacheLine* Cache::getCacheLine(const void *tag)
{
    CacheLine *cacheLine = nullptr;
    if (tag == nullptr) return cacheLine;

    const int setIndex = getSetIndex(tag);
    const int offset = setIndex * this->nrOfLinesPerSet;
    const int end = offset + this->nrOfLinesPerSet;
    for (int idx = offset; idx < end; ++idx) 
    {
        if (this->lines[idx].tag == tag)
        {
            cacheLine = &this->lines[idx];

            if (this->replacementPolicy == LRU) 
                this->lines[idx].age = 0;
            else
                break;
        } 
        
        if (this->replacementPolicy == LRU) 
        {
            this->lines[idx].age += 1;
        }
    }

    return cacheLine;
}

/*!
    * \brief decide which line to replace and do so
    * \param tag The tag of the new line
    * \param size The size of the data
    * \param deviceAddress The device address of the new line
    * \param flag Indicates whether the most recent data is on the CPU, GPU or both
    * \param idx (optional) provide the index of the cache line to be updated, if not provided it'll use the replacement policy to determine the index
    * \return Returns the cache line
    */
CacheLine Cache::addToCache(const void *tag, size_t size, cl_mem deviceAddress, Flag flag, int idx)
{
    if (idx == -1) 
    {
        idx = getSetIndex(tag);
        if (this->organisation != DIRECT_MAPPING)
        {
            // Replacement policy needed
            switch (this->replacementPolicy) 
            {
                case LRU:
                    // Get the oldest index
                    idx = getOldestIndex(idx);
                    dout << "LRU idx: " << idx << endl;
                    break;
                case FIFO: {
                        int setIndex = idx;
                        int maxIter = 1000;
                        do {
                            this->FIFO_index[setIndex] = (this->FIFO_index[setIndex] + 1) % this->nrOfLinesPerSet;
                            idx = this->FIFO_index[setIndex] + setIndex * this->nrOfLinesPerSet;
                            // dout << "fifo idx: " << idx << endl;
                        } while (find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end() && --maxIter > 0);
                        if (maxIter == 0) {
                            cout << "Can't replace line using FIFO, infinite loop" << endl;
                            for (auto x : this->lockedLines) {
                                cout << x << endl;
                            }
                            printCache();
                            exit(1);
                        }
                    }
                    break;
                case RANDOM:
                    idx = getRandomIndex(idx);
                    dout << "random idx: " << idx << endl;
                    break;
                case SMALLEST:
                    idx = getSmallestDataLine(idx);
                    dout << "smallest idx: " << idx << endl;
                    break;
                default:
                    dout << "Replacement policy not implemented" << endl;
                    break;
            }
        } 
        else if (find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end()) 
        {
            int randomInt;
            do {
                randomInt = rand();
                idx = randomInt % this->nrOfLines;
            } // Repeat until we find an index that is not locked
            while(find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end());
        }
        
    }
    // cout << "write back: " << write_back << endl;
    if (this->write_back && this->lines[idx].flag == GPU) 
    {
        dout << "Replacing cache line and writing back" << endl;
        // Write back to device
        cl_event myevent;
        
        clEnqueueReadBuffer(this->cache_command_queue, this->lines[idx].deviceAddress, CL_TRUE, 0, this->lines[idx].size, this->lines[idx].tag, 0, NULL, &myevent);
        this->duration.bytesSaved -= this->lines[idx].size;

        this->duration.deviceToHost += probe_event_time(myevent, this->cache_command_queue);
        this->lines[idx].flag = BOTH;        
    } 

    if (this->lines[idx].deviceAddress != deviceAddress 
        && this->lines[idx].deviceAddress != nullptr)
    {
        // There is an old buffer on this cache line, so free that first to avoid memory leaks.
        buffers--;
        clReleaseMemObject(this->lines[idx].deviceAddress);
    }
    
    this->lockedLines.push_back(idx);
    this->lines[idx].flag = flag;
    this->lines[idx].age = 0;
    this->lines[idx].tag = (void*) tag;
    this->lines[idx].size = size;
    this->lines[idx].deviceAddress = deviceAddress;
    return this->lines[idx];
}

/*! 
    * \brief Get a random index within the set and return the cache index.
    * \param idx The set index
    * \return The random cache index
    */
int Cache::getRandomIndex(int setIndex) 
{   
    int randomInt, idx, maxIter = 1000;
    do {
        randomInt = rand();
        idx = setIndex * nrOfLinesPerSet + randomInt % nrOfLinesPerSet;
    } // Repeat until we find an index that is not locked
    while(find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end() && --maxIter > 0);

    if (maxIter == 0) {
        cout << "Can't replace line using random, infinite loop" << endl;
        for (auto x : this->lockedLines) {
            cout << x << endl;
        }
        printCache();
        exit(1);
    }

    return idx;
}

int Cache::getOldestIndex(int setIndex, bool increaseAge)
{
    int oldestLineIndex = -1;
    int oldestLineAge = -1;

    const int offset = setIndex * this->nrOfLinesPerSet;
    const int end = offset + this->nrOfLinesPerSet;
    for (int idx = offset; idx < end; ++idx)
    {
        if (this->lines[idx].age > oldestLineAge)
        {
            // Make sure that this line is not locked
            if (find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end()) 
                continue;

            oldestLineAge = this->lines[idx].age;
            oldestLineIndex = idx;
        }

        if (increaseAge) 
            this->lines[idx].age += 1;
    }

    if (oldestLineIndex == -1) {
        dout << "Can't replace line using FIFO/LRU, replace a random line" << endl;
        oldestLineIndex = getRandomIndex(setIndex);
    }
    return oldestLineIndex;
}

int Cache::getSmallestDataLine(int setIndex)
{
    int smallestLineIndex = -1;
    int smallestLineSize = -1;

    const int offset = setIndex * this->nrOfLinesPerSet;
    const int end = offset + this->nrOfLinesPerSet;
    for (int idx = offset; idx < end; ++idx)
    {
        if (this->lines[idx].size < smallestLineSize)
        {
            // Make sure that this line is not locked
            if (find(this->lockedLines.begin(), this->lockedLines.end(), idx) != this->lockedLines.end()) 
                continue;

            smallestLineSize = this->lines[idx].size;
            smallestLineIndex = idx;
        }
    }

    if (smallestLineIndex == -1) {
        cout << "Can't replace line using FIFO/LRU, replace a random line" << endl;
        smallestLineIndex = getRandomIndex(setIndex);
    }
    return smallestLineIndex;
}
